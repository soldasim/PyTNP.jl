function _setup_python_path!()
	sys = pyimport("sys")
	src_path = abspath(@__DIR__)
	if pyconvert(Bool, src_path ∉ sys.path)
		sys.path.insert(0, src_path)
	end
end

abstract type TNPType end
struct StandardTNP <: TNPType end

"""
	TNPModel

Handle for a loaded Transformer Neural Process model and its device.
"""
struct TNPModel{T<:TNPType}
    mode::T
	model::Py
	device::String
end

"""
	init_model(; x_dim::Int = 1,
			y_dim::Int = 1,
			dim_model::Int = 128,
			embedder_depth::Int = 2,
			predictor_depth::Int = 2,
			num_heads::Int = 4,
			encoder_depth::Int = 2,
			dim_feedforward::Int = 512,
			dropout::Float64 = 0.1,
			device::Union{Nothing, String} = nothing)

Initialize a Transformer Neural Process model.

# Arguments
- Model architecture parameters (must match training configuration)
- `device`: Optional device override ("mps", "cuda", or "cpu")

# Example
```julia
model = init_model()
```
"""
function init_model(;
    x_dim::Int = 1,
    y_dim::Int = 1,
    dim_model::Int = 64,
    embedder_depth::Int = 4,
    predictor_depth::Int = 2,
    num_heads::Int = 8,
    encoder_depth::Int = 6,
    dim_feedforward::Int = 2 * dim_model,
    dropout::Float64 = 0.0,
    device::Union{Nothing, String} = nothing,
    mode::TNPType = StandardTNP(),
)
	# Import Python modules
	torch = pyimport("torch")
	tnp_module = pyimport("tnp")
	initialize_tnp = tnp_module.initialize_tnp

	# Determine device
	if device === nothing
		device = if pyconvert(Bool, torch.backends.mps.is_available())
			"mps"
		elseif pyconvert(Bool, torch.cuda.is_available())
			"cuda"
		else
			"cpu"
		end
	end

	py_model = initialize_tnp(
		x_dim = x_dim,
		y_dim = y_dim,
		dim_model = dim_model,
		embedder_depth = embedder_depth,
		predictor_depth = predictor_depth,
		num_heads = num_heads,
		encoder_depth = encoder_depth,
		dim_feedforward = dim_feedforward,
		dropout = dropout,
		device = device
	)

	return TNPModel(mode, py_model, device)
end

"""
	load_model(model_path::String = "tnp_model.pt";
			   device::Union{Nothing, String} = nothing)

Load the trained TNP model from a file.

# Arguments
- `model_path::String`: Path to the saved model weights
- `device`: Optional device override ("mps", "cuda", or "cpu")

# Example
```julia
model = load_model("tnp_model.pt")
```
"""
function load_model(model_path::String = "tnp_model.pt";
    device::Union{Nothing, String} = nothing,
    mode::TNPType = StandardTNP(),
)
	# Import Python modules
	torch = pyimport("torch")
	weights = pyimport("weights")

	# Determine device
	if device === nothing
		device = if pyconvert(Bool, torch.backends.mps.is_available())
			"mps"
		elseif pyconvert(Bool, torch.cuda.is_available())
			"cuda"
		else
			"cpu"
		end
	end

	py_model = weights.load_model(model_path, device=device)
	println("Model loaded successfully on device: $(device)")
	return TNPModel(mode, py_model, device)
end

"""
	save_model(model::TNPModel, model_path::String = "tnp_model.pt")

Save model weights and hyperparameters to a file.

# Arguments
- `model`: TNP model handle to save
- `model_path::String`: Path to save the model weights
"""
function save_model(model::TNPModel, model_path::String = "tnp_model.pt")
	weights = pyimport("weights")
	weights.save_model(model.model, model_path)
end

"""
	predict(model::TNPModel, context_x, context_y, target_x)

Make predictions with the TNP model.

# Arguments
- `context_x`: Context input locations (Vector or Matrix of size N×x_dim)
- `context_y`: Context output values (Vector or Matrix of size N×y_dim)
- `target_x`: Target input locations (Vector or Matrix of size M×x_dim)

# Returns
- `mean`: Predicted means (Vector if y_dim=1, Matrix of size M×y_dim otherwise)
- `std`: Predicted standard deviations (Vector if y_dim=1, Matrix of size M×y_dim otherwise)

# Example (1D)
```julia
using PyTNP

# Load model first
model = load_model("tnp_model.pt")

# Define context and target points
context_x = [-1.5, -1.0, -0.5, 0.0, 0.5]
context_y = [1.2, 0.8, 0.3, 0.1, 0.4]
target_x = range(-2, 2, length=50) |> collect

# Get predictions
mean, std = predict(model, context_x, context_y, target_x)

# Plot results
using Plots
plot(target_x, mean, ribbon=2*std, label="Prediction ±2σ")
scatter!(context_x, context_y, label="Context", markersize=8)
```

# Example (2D inputs, 1D outputs)
```julia
# 2D inputs
context_x = rand(20, 2)  # 20 points in 2D space
context_y = sin.(context_x[:, 1]) .* cos.(context_x[:, 2])  # 1D outputs
target_x = rand(100, 2)  # 100 test points

# Get predictions
mean, std = predict(model, context_x, context_y, target_x)
```
"""
function predict(model::TNPModel, context_x, context_y, target_x)
    return _predict(model.mode, model, context_x, context_y, target_x)
end

# Convert 1D vector to 2D matrix
function _predict(
    mode::TNPType,
    model::TNPModel,
    context_x::AbstractMatrix{<:Real}, 
    context_y::AbstractMatrix{<:Real}, 
    target_x::AbstractVector{<:Real},
)    
    return _predict(mode, model, context_x, context_y, hcat(target_x))
end

# Convert 2D matrices to 3D arrays
function _predict(
    mode::TNPType,
    model::TNPModel,
    context_x::AbstractMatrix{<:Real}, 
    context_y::AbstractMatrix{<:Real}, 
    target_x::AbstractMatrix{<:Real},
)
    # Reshape from (dim, N) to (dim, N, 1) for single batch
    context_x_3d = reshape(context_x, size(context_x)..., 1)
    context_y_3d = reshape(context_y, size(context_y)..., 1)
    target_x_3d = reshape(target_x, size(target_x)..., 1)
    
    return _predict(mode, model, context_x_3d, context_y_3d, target_x_3d)
end

# The `StandardTNP` prediction mode
function _predict(
    ::StandardTNP,
    model::TNPModel,
    context_x::AbstractArray{<:Real, 3}, 
    context_y::AbstractArray{<:Real, 3}, 
    target_x::AbstractArray{<:Real, 3},
)
	torch = pyimport("torch")
	np = pyimport("numpy")
	
	# Helper to convert Julia 3D array to PyTorch tensor
	# Julia is column-major, NumPy/PyTorch is row-major
	# We need to permute: (dim, N, batch) -> (batch, N, dim)
	function to_tensor(data::AbstractArray{<:Real, 3})
		arr = permutedims(data, (3, 2, 1))
		np_arr = np.asarray(arr, dtype=np.float32)
		tensor = torch.from_numpy(np_arr).to(model.device)
		return tensor
	end
	
	# Convert to PyTorch tensors [batch, N, dim]
	context_x_tensor = to_tensor(context_x)
	context_y_tensor = to_tensor(context_y)
	target_x_tensor = to_tensor(target_x)
	
	# Make predictions with no gradient
	with_no_grad = torch.no_grad()
	with_no_grad.__enter__()
	
	try
		pred_mean, pred_std = model.model(context_x_tensor, context_y_tensor, target_x_tensor)
		
		# Convert back to Julia arrays
		# pred_mean and pred_std have shape [1, M, y_dim]
		y_dim = pyconvert(Int, pred_mean.shape[2])
		
		if y_dim == 1
			# Return as vectors for y_dim=1
			mean = pyconvert(Vector{Float64}, pred_mean[0, pybuiltins.slice(nothing), 0].cpu().numpy())
			std = pyconvert(Vector{Float64}, pred_std[0, pybuiltins.slice(nothing), 0].cpu().numpy())
		else
			# Return as matrices for y_dim>1
			mean_np = pyconvert(Array{Float64}, pred_mean[0, pybuiltins.slice(nothing), pybuiltins.slice(nothing)].cpu().numpy())
			std_np = pyconvert(Array{Float64}, pred_std[0, pybuiltins.slice(nothing), pybuiltins.slice(nothing)].cpu().numpy())
			mean = mean_np'
			std = std_np'
		end
		
		return mean, std
	finally
		with_no_grad.__exit__(nothing, nothing, nothing)
	end
end

"""
	train_model!(model::TNPModel, sample_fn::Union{Py, Function};
			model_path::Union{Nothing, String} = nothing,
			save_path::Union{Nothing, String} = "data/tnp_model.pt",
			num_iterations::Int = 10000,
			lr_start::Float64 = 1e-4,
			lr_end::Float64 = 1e-6,
			print_freq::Int = 1000,
			device::Union{Nothing, String} = nothing)

Train the Transformer Neural Process model using the Python training loop.

# Arguments
- `model`: TNP model handle to train
- `sample_fn`: Python callable or Julia function that returns (context_x, context_y, target_x, target_y)
- `model_path`: Optional path to initial weights to load before training
- `save_path`: Optional path to save the trained weights
- `lr_start`: Starting learning rate for cosine annealing
- `lr_end`: Ending learning rate for cosine annealing
- Training parameters (see `train.py`)

# Returns
- `Vector{Float64}`: Learning rates over training iterations
- `Vector{Float64}`: Loss values over training iterations
- `Vector{Float64}`: Rolling average loss values over training iterations
"""
function train_model!(model::TNPModel, sample_fn::Union{Py, Function};
    model_path::Union{Nothing, String} = nothing,
    save_path::Union{Nothing, String} = "data/tnp_model.pt",
    num_iterations::Int = 10_000,
    lr_start::Float64 = 5e-4,
    lr_end::Float64 = 0.0,
    warmup_ratio::Float64 = 0.05,
    start_factor::Float64 = 0.05,
    rolling_window::Int = 100,
    print_freq::Int = 500,
    device::Union{Nothing, String} = nothing,
)	
	# Import training function
	train_module = pyimport("train")
	train_tnp = train_module.train_tnp

	py_sample_fn = _to_py_callable(sample_fn)

	# Train model
	learning_rates, losses, rolling_avg_losses = train_tnp(model.model, py_sample_fn,
		num_iterations = num_iterations,
		lr_start = lr_start,
		lr_end = lr_end,
		warmup_ratio = warmup_ratio,
		start_factor = start_factor,
		rolling_window = rolling_window,
		print_freq = print_freq,
		device = device,
		model_path = model_path,
		save_path = save_path
	)

	return pyconvert(Vector{Float64}, learning_rates),
            pyconvert(Vector{Float64}, losses),
            pyconvert(Vector{Float64}, rolling_avg_losses)
end

_to_py_callable(sample_fn::Py) = sample_fn
_to_py_callable(sample_fn::Function) = Py(sample_fn)
