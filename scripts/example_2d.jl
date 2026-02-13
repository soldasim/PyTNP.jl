"""
Example usage of the PyTNP Julia interface with 2D inputs.

This script trains (or loads) a model, evaluates it on a simple
synthetic function with 2D inputs and 1D outputs, and plots results from Julia only.
"""

# This has to be done before loading PyTNP.
ENV["JULIA_PYTHONCALL_EXE"] = "/Users/soldasim/Documents/julia-pkg/PyTNP.jl/venv/bin/python"

using PyTNP
using PythonCall
using CairoMakie
using Random
using Statistics

root_dir = normpath(joinpath(@__DIR__, ".."))
data_dir = joinpath(root_dir, "data")
mkpath(data_dir)

model_path = joinpath(data_dir, "tnp_model_2d.pt")
loss_plot_path = joinpath(data_dir, "tnp_training_loss_2d.png")
pred_plot_path = joinpath(data_dir, "tnp_predictions_2d.png")

if isfile(model_path)
    println("Loading TNP model from $model_path ...")
    model = load_model(model_path)
else
    println("No model found. Training a new model...")
    gp_sampler = pyimport("gp_sampler")
    
    model = init_model(
        x_dim = 2,
        y_dim = 1,
        dim_model = 128,
        num_heads = 4,
        encoder_depth = 2
    )

    sample_fn = gp_sampler.make_gp_sampler(
        batch_size = 16,
        num_context_range = (1, 99),
        num_total_points = 100,
        x_range = (-2.0, 2.0),
        kernel_length_scale = 1.0,
        kernel_variance = 1.0,
        noise_variance = 1e-8,
        x_dim = pyconvert(Int, model.model.x_dim),
        y_dim = pyconvert(Int, model.model.y_dim)
    )

    _, losses, avg_losses = train_model!(
        model,
        sample_fn;
        num_iterations = 2000,
        print_freq = 100,
        save_path = model_path,
        device = model.device,
    )

    fig_loss = Figure()
    ax_loss = Axis(
        fig_loss[1, 1],
        xlabel = "Iteration",
        ylabel = "Loss",
        title = "Training Loss"
    )
    lines!(ax_loss, 1:length(losses), losses, linewidth = 1, alpha = 0.3, label = "Loss")
    lines!(ax_loss, 1:length(avg_losses), avg_losses, linewidth = 2, color = :red, label = "Rolling Average")
    axislegend(ax_loss, position = :rt)
    save(loss_plot_path, fig_loss)
    println("Training loss plot saved to $loss_plot_path")
end

println("Model loaded on device: $(model.device)")

Random.seed!(42)
# 2D function: f(x1, x2) = sin(x1) * cos(x2)
f(x1, x2) = sin(x1) * cos(x2)

# Create context points: random samples in 2D space
n_context = 50
context_x1 = rand(n_context) .* 4.0 .- 2.0  # Random points in [-2, 2]
context_x2 = rand(n_context) .* 4.0 .- 2.0
context_x = hcat(context_x1, context_x2)  # Shape: (n_context, 2)
context_y = f.(context_x1, context_x2)

# Create target points: grid in 2D space
grid_size = 50
target_x1 = repeat(range(-2.0, 2.0, length = grid_size), grid_size)
target_x2 = repeat(range(-2.0, 2.0, length = grid_size), inner = grid_size)
target_x = hcat(target_x1, target_x2)  # Shape: (grid_size^2, 2)
target_y = f.(target_x1, target_x2)

println("\nMaking predictions...")
pred_mean, pred_std = predict(model, context_x', context_y', target_x')

mse = mean((pred_mean .- target_y) .^ 2)
println("Evaluation MSE: $mse")

# Reshape for plotting
pred_mean_grid = reshape(pred_mean, grid_size, grid_size)
pred_std_grid = reshape(pred_std, grid_size, grid_size)
target_y_grid = reshape(target_y, grid_size, grid_size)

fig = Figure(size = (1200, 400))

# Plot 1: True function
ax1 = Axis(
    fig[1, 1],
    xlabel = "x1",
    ylabel = "x2",
    title = "True Function",
    aspect = DataAspect()
)
hm1 = heatmap!(ax1, range(-2.0, 2.0, length = grid_size), range(-2.0, 2.0, length = grid_size), target_y_grid, colormap = :viridis)
scatter!(ax1, context_x1, context_x2, color = :red, markersize = 10, strokewidth = 2, strokecolor = :white, label = "Context")
Colorbar(fig[1, 2], hm1)

# Plot 2: Prediction mean
ax2 = Axis(
    fig[1, 3],
    xlabel = "x1",
    ylabel = "x2",
    title = "TNP Prediction Mean",
    aspect = DataAspect()
)
hm2 = heatmap!(ax2, range(-2.0, 2.0, length = grid_size), range(-2.0, 2.0, length = grid_size), pred_mean_grid, colormap = :viridis)
scatter!(ax2, context_x1, context_x2, color = :red, markersize = 10, strokewidth = 2, strokecolor = :white, label = "Context")
Colorbar(fig[1, 4], hm2)

# Plot 3: Prediction uncertainty
ax3 = Axis(
    fig[1, 5],
    xlabel = "x1",
    ylabel = "x2",
    title = "TNP Prediction Std",
    aspect = DataAspect()
)
hm3 = heatmap!(ax3, range(-2.0, 2.0, length = grid_size), range(-2.0, 2.0, length = grid_size), pred_std_grid, colormap = :plasma)
scatter!(ax3, context_x1, context_x2, color = :red, markersize = 10, strokewidth = 2, strokecolor = :white, label = "Context")
Colorbar(fig[1, 6], hm3)

save(pred_plot_path, fig)
println("Prediction plot saved to $pred_plot_path")

display(fig)
