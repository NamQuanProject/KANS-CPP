#include "KANsLinear.h"
#include <torch/torch.h>
#include <cmath>

KANLinearImpl::KANLinearImpl(
    int64_t in_features,
    int64_t out_features,
    int64_t grid_size,
    int64_t spline_order,
    double scale_noise,
    double scale_base,
    double scale_spline,
    bool enable_standalone_scale_spline,
    torch::nn::SiLU base_activation,
    double grid_eps,
    std::pair<double, double> grid_range

) : in_features(in_features), out_features(out_features), grid_size(grid_size),
    spline_order(spline_order), scale_noise(scale_noise), scale_base(scale_base),
    scale_spline(scale_spline), enable_standalone_scale_spline(enable_standalone_scale_spline),
    base_activation(base_activation), grid_eps(grid_eps) {
    
    double h = (grid_range.second - grid_range.first) / grid_size;
    grid = (torch::arange(-spline_order, grid_size + spline_order + 1) * h + grid_range.first)
                .expand({in_features, -1})
                .contiguous();
    register_buffer("grid", grid);

    base_weight = register_parameter("base_weight", torch::empty({out_features, in_features}));
    spline_weight = register_parameter("spline_weight", torch::empty({out_features, in_features, grid_size + spline_order}));
    if (enable_standalone_scale_spline) {
        spline_scaler = register_parameter("spline_scaler", torch::empty({out_features, in_features}));
    }

    reset_parameters();
}


void KANLinearImpl::reset_parameters() {
    torch::nn::init::kaiming_uniform_({base_weight}, std::sqrt(5) * scale_base);

    torch::NoGradGuard no_grad;
    auto noise = (torch::rand({grid_size + 1, in_features, out_features}) - 0.5) * scale_noise / grid_size;
    
    // Get x tensor for curve2coeff
    auto x_for_coeff = grid.index({torch::indexing::Slice(spline_order, -spline_order)});
    
    // Print shapes for debugging
    std::cout << "x shape: " << x_for_coeff.sizes() << std::endl;
    std::cout << "y shape: " << noise.sizes() << std::endl;

    // Apply scaling factor to spline weights
    double scale_factor = enable_standalone_scale_spline ? 1.0 : scale_spline;
    spline_weight.copy_(scale_factor * curve2coeff(x_for_coeff, noise));

    // Initialize spline scaler if enabled
    if (enable_standalone_scale_spline) {
        torch::nn::init::kaiming_uniform_(spline_scaler, std::sqrt(5) * scale_spline);
    }
}



torch::Tensor KANLinearImpl::b_splines(torch::Tensor x) {
    assert(x.dim() == 2 && x.size(1) == in_features);

    auto grid_t = grid;
    x = x.unsqueeze(-1);
    auto bases = ((x >= grid_t.slice(1, 0, -1)) & (x < grid_t.slice(1, 1))).to(x.dtype());

    for (int k = 1; k <= spline_order; k++) {
        bases = (x - grid_t.slice(1, 0, -(k + 1))) / (grid_t.slice(1, k, -1) - grid_t.slice(1, 0, -(k + 1))) * bases.slice(2, 0, -1) +
                (grid_t.slice(1, k + 1) - x) / (grid_t.slice(1, k + 1) - grid_t.slice(1, 1, -k)) * bases.slice(2, 1);
    }

    assert(bases.size(0) == x.size(0) && bases.size(1) == in_features && bases.size(2) == grid_size + spline_order);
    return bases.contiguous();
}

torch::Tensor KANLinearImpl::curve2coeff(torch::Tensor x, torch::Tensor y) {
   
    
    assert(x.dim() == 2 && x.size(1) == in_features);
    std::cout << "curve2coeff: x shape: " << x.sizes() << ", y shape: " << y.sizes() << std::endl;

    assert(y.size(0) == x.size(0) && y.size(1) == in_features && y.size(2) == out_features);
    auto A = b_splines(x).transpose(0, 1);
    auto B = y.transpose(0, 1);
    auto A_pinv = torch::pinverse(A);
    auto solution = torch::matmul(A_pinv, B);
    auto result = solution.permute({2, 0, 1});
    assert(result.size(0) == out_features && result.size(1) == in_features && result.size(2) == grid_size + spline_order);
    return result.contiguous();
}

torch::Tensor KANLinearImpl::scaled_spline_weight() {
    if (enable_standalone_scale_spline) {
        return spline_weight * spline_scaler.unsqueeze(-1);
    }
    return spline_weight;
}

torch::Tensor KANLinearImpl::forward(torch::Tensor x) {
    assert(x.size(-1) == in_features);
    auto original_shape = x.sizes();
    x = x.view({-1, in_features});

    auto base_output = torch::nn::functional::linear(base_activation(x), base_weight);
    auto spline_output = torch::nn::functional::linear(b_splines(x).view({x.size(0), -1}), scaled_spline_weight().view({out_features, -1}));
    auto output = base_output + spline_output;

    // output = output.view(original_shape.slice(0, original_shape.size() - 1).append(out_features));
    return output;
}

void KANLinearImpl::update_grid(torch::Tensor x, double margin) {
    assert(x.dim() == 2 && x.size(1) == in_features);
    auto batch = x.size(0);

    auto splines = b_splines(x).permute({1, 0, 2});
    auto orig_coeff = scaled_spline_weight().permute({1, 2, 0});
    auto unreduced_spline_output = torch::bmm(splines, orig_coeff).permute({1, 0, 2});

    auto x_sorted = std::get<0>(torch::sort(x, 0));
    auto grid_adaptive = x_sorted.index_select(0, torch::linspace(0, batch - 1, grid_size + 1, torch::TensorOptions().dtype(torch::kInt64)));

    auto uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / grid_size;
    auto grid_uniform = torch::arange(grid_size + 1, torch::TensorOptions().dtype(torch::kFloat32)).unsqueeze(1) * uniform_step + x_sorted[0] - margin;

    auto grid = grid_eps * grid_uniform + (1 - grid_eps) * grid_adaptive;
    grid = torch::cat({
        grid.slice(0, 0, 1) - uniform_step * torch::arange(spline_order, 0, -1, torch::TensorOptions().device(x.device())).unsqueeze(1),
        grid,
        grid.slice(0, -1) + uniform_step * torch::arange(1, spline_order + 1, torch::TensorOptions().device(x.device())).unsqueeze(1)
    }, 0);

    this->grid.copy_(grid.transpose(0, 1));
    spline_weight.copy_(curve2coeff(x, unreduced_spline_output));
}

torch::Tensor KANLinearImpl::regularization_loss(double regularize_activation, double regularize_entropy) {
    auto l1_fake = spline_weight.abs().mean(-1);
    auto regularization_loss_activation = l1_fake.sum();
    auto p = l1_fake / regularization_loss_activation;
    auto regularization_loss_entropy = -(p * p.log()).sum();
    return regularize_activation * regularization_loss_activation + regularize_entropy * regularization_loss_entropy;
}
