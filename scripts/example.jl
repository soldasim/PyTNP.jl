"""
Example usage of the PyTNP Julia interface.

This script trains (or loads) a model, evaluates it on a simple
synthetic function, and plots results from Julia only.
"""

# This has to be done before loading PyTNP.
ENV["JULIA_PYTHONCALL_EXE"] = "/Users/soldasim/Documents/julia-pkg/PyTNP.jl/venv/bin/python"

using PyTNP
using CairoMakie
using Random
using Statistics

root_dir = normpath(joinpath(@__DIR__, ".."))
data_dir = joinpath(root_dir, "data")
mkpath(data_dir)

model_path = joinpath(data_dir, "tnp_model.pt")
loss_plot_path = joinpath(data_dir, "tnp_training_loss.png")
pred_plot_path = joinpath(data_dir, "tnp_julia_predictions.png")

if isfile(model_path)
    println("Loading TNP model from $model_path ...")
    model = PyTNP.load_model(model_path)
else
    println("No model found. Training a new model...")
    losses = PyTNP.train_model(
        num_iterations = 2000,
        print_freq = 200,
        save_path = model_path
    )

    fig_loss = Figure()
    ax_loss = Axis(
        fig_loss[1, 1],
        xlabel = "Iteration",
        ylabel = "Loss",
        title = "Training Loss"
    )
    lines!(ax_loss, 1:length(losses), losses, linewidth = 2)
    save(loss_plot_path, fig_loss)
    println("Training loss plot saved to $loss_plot_path")

    model = PyTNP.load_model(model_path)
end

println("Model loaded on device: $(model.device)")

Random.seed!(42)
f(x) = sin(x)

context_x = collect(range(-2.0, 2.0, length = 12))
context_y = f.(context_x) .+ 0.05 .* randn(length(context_x))

target_x = collect(range(-2.0, 2.0, length = 200))
target_y = f.(target_x)

println("\nMaking predictions...")
pred_mean, pred_std = PyTNP.predict(model, context_x, context_y, target_x)

mse = mean((pred_mean .- target_y) .^ 2)
println("Evaluation MSE: $mse")

fig = Figure()
ax = Axis(
    fig[1, 1],
    xlabel = "x",
    ylabel = "y",
    title = "TNP Predictions from Julia"
)

band!(
    ax,
    target_x,
    pred_mean .- 2 .* pred_std,
    pred_mean .+ 2 .* pred_std,
    color = (:red, 0.25),
    label = "Prediction +/- 2 std"
)
lines!(ax, target_x, pred_mean, linewidth = 2, color = :red)
lines!(ax, target_x, target_y, linewidth = 2, color = :black, label = "True function")
scatter!(
    ax,
    context_x,
    context_y,
    markersize = 10,
    strokewidth = 1.5,
    color = :blue,
    label = "Context points"
)
axislegend(ax, position = :lt)

save(pred_plot_path, fig)
println("Prediction plot saved to $pred_plot_path")

display(fig)
