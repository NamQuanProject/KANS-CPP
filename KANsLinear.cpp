#include "KANsLinear.h"

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
    torch::nn::init::kaiming_uniform_(base_weight, std::sqrt(5) * scale_base);

    torch::NoGradGuard no_grad;

    auto noise = (torch::rand({grid_size + 1, in_features, out_features}) - 0.5) * scale_noise / grid_size;

    auto grid_t = grid.transpose(0, 1);

    auto sliced_grid = grid_t.index({torch::indexing::Slice(spline_order, -spline_order)});

    spline_weight.copy_(
        (enable_standalone_scale_spline ? 1.0 : scale_spline) * curve2coeff(sliced_grid, noise)
    );

    if (enable_standalone_scale_spline) {
        torch::nn::init::kaiming_uniform_(spline_scaler, std::sqrt(5) * scale_spline);
    }
}





torch::Tensor KANLinearImpl::b_splines(torch::Tensor x) {
    assert(x.dim() == 2 && x.size(1) == in_features);

    auto grid_t = grid;
    x = x.unsqueeze(-1);

    auto bases = ((x >= grid_t.slice(1, 0, -1)) & (x < grid_t.slice(1, 1))).to(x.dtype());

    /*
    N_{i,k}(x) = ((x - t_i) / (t_{i+k} - t_i)) * N_{i,k-1}(x) 
                + ((t_{i+k+1} - x) / (t_{i+k+1} - t_{i+1})) * N_{i+1,k-1}(x)

    Where:
    - N_{i,k}(x) is the B-spline basis function of degree k.
    - t_i are the knots (grid points).
    - x is the input value.
    - The formula recursively combines the lower-order basis functions to build the higher-order B-spline.
    */
    for (int k = 1; k <= spline_order; k++) {
        bases = (x - grid_t.slice(1, 0, -(k + 1))) / (grid_t.slice(1, k, -1) - grid_t.slice(1, 0, -(k + 1))) * bases.slice(2, 0, -1) +
                (grid_t.slice(1, k + 1) - x) / (grid_t.slice(1, k + 1) - grid_t.slice(1, 1, -k)) * bases.slice(2, 1);
    }
    
    assert(bases.size(0) == x.size(0) && bases.size(1) == in_features && bases.size(2) == grid_size + spline_order);
    return bases.contiguous();
}


torch::Tensor KANLinearImpl::curve2coeff(torch::Tensor x, torch::Tensor y) {
    /*
    Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
    */

    torch::Tensor A = b_splines(x).transpose(0, 1);  
    // (in_features, batch_size, grid_size + spline_order)
    torch::Tensor B = y.transpose(0, 1);  
    // (in_features, batch_size, out_features)

    auto solution = torch::linalg::lstsq(A, B, c10::nullopt, "gels");
    torch::Tensor result = std::get<0>(solution).permute({2, 0, 1}); 
    // (out_features, in_features, grid_size + spline_order)

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
    assert(x.dim() == 2 && x.size(1) == in_features);
    auto original_shape = x.sizes();
    x = x.view({-1, in_features});

    auto base_output = torch::nn::functional::linear(base_activation(x), base_weight);
    auto spline_output = torch::nn::functional::linear(b_splines(x).view({x.size(0), -1}), scaled_spline_weight().view({out_features, -1}));
    auto output = base_output + spline_output;

    return output;
}

void KANLinearImpl::update_grid(torch::Tensor x, double margin) {
    // GRID EXTENSION METHODS
    assert(x.dim() == 2 && x.size(1) == in_features);
    int64_t batch = x.size(0);

    auto splines = b_splines(x);  
    splines = splines.permute({1, 0, 2});  

    auto orig_coeff = scaled_spline_weight();  
    orig_coeff = orig_coeff.permute({1, 2, 0});  

    // CALCULATE THE BATCH MULTIPLICATION
    auto unreduced_spline_output = torch::bmm(splines, orig_coeff);  
    unreduced_spline_output = unreduced_spline_output.permute({1, 0, 2});

    auto sorted_result = torch::sort(x, 0);
    auto x_sorted = std::get<0>(sorted_result);
    
    auto indices = torch::linspace(0, batch - 1, grid_size + 1, torch::TensorOptions().dtype(torch::kLong).device(x.device()));

    // Adaptive Grid Computation: 
    auto grid_adaptive = x_sorted.index_select(0, indices);
    // Uniform Grid Creation:
    auto uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / grid_size;
    auto grid_uniform = torch::arange(grid_size + 1, torch::TensorOptions().dtype(torch::kFloat32).device(x.device()))
                        .unsqueeze(1)
                        .mul(uniform_step)
                        .add(x_sorted[0] - margin);
    
    grid = grid_eps * grid_uniform + (1 - grid_eps) * grid_adaptive;
    
    auto lower_extension = grid.slice(0, 0, 1).clone() - uniform_step * torch::arange(spline_order, 0, -1, torch::TensorOptions().device(x.device())).unsqueeze(1);
    auto upper_extension = grid.slice(0, -1).clone() + uniform_step * torch::arange(1, spline_order + 1, torch::TensorOptions().device(x.device())).unsqueeze(1);

    grid = torch::concatenate({lower_extension, grid, upper_extension}, 0);
    
    auto cat_grid = grid.clone();
    grid = cat_grid.transpose(0, 1).clone();

    
    auto coeff = curve2coeff(x, unreduced_spline_output);
    spline_weight.data().copy_(coeff);
}




torch::Tensor KANLinearImpl::regularization_loss(double regularize_activation, double regularize_entropy) {
    /*
    Compute the regularization loss.
        This is a simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
    */
    
    auto l1_fake = spline_weight.abs().mean(-1);
    
    auto regularization_loss_activation = l1_fake.sum();

    auto p = l1_fake / regularization_loss_activation;

    auto regularization_loss_entropy = -(p * p.log()).sum();

    return (
    regularize_activation * regularization_loss_activation 
    + regularize_entropy * regularization_loss_entropy
    );
}
