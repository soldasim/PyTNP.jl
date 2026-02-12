function _setup_python_path!()
	sys = pyimport("sys")
	src_path = abspath(@__DIR__)
	if pyconvert(Bool, src_path ∉ sys.path)
		sys.path.insert(0, src_path)
	end
end

"""
	TNPModel

Handle for a loaded Transformer Neural Process model and its device.
"""
struct TNPModel
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
function init_model(; x_dim::Int = 1,
			y_dim::Int = 1,
			dim_model::Int = 128,
			embedder_depth::Int = 2,
			predictor_depth::Int = 2,
			num_heads::Int = 4,
			encoder_depth::Int = 2,
			dim_feedforward::Int = 512,
			dropout::Float64 = 0.1,
			device::Union{Nothing, String} = nothing)
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

	return TNPModel(py_model, device)
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
					device::Union{Nothing, String} = nothing)
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
	return TNPModel(py_model, device)
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
	predict(model::TNPModel, context_x::Vector{<:Real}, context_y::Vector{<:Real}, target_x::Vector{<:Real})

Make predictions with the TNP model.

# Arguments
- `context_x::Vector{<:Real}`: Context input locations
- `context_y::Vector{<:Real}`: Context output values
- `target_x::Vector{<:Real}`: Target input locations

# Returns
- `mean::Vector{Float64}`: Predicted means
- `std::Vector{Float64}`: Predicted standard deviations

# Example
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
"""
function predict(model::TNPModel, context_x::Vector{<:Real}, context_y::Vector{<:Real}, target_x::Vector{<:Real})
	torch = pyimport("torch")
	np = pyimport("numpy")
	
	# Convert to PyTorch tensors [1, N, 1]
	context_x_arr = Array(reshape(Float32.(collect(context_x)), 1, :, 1))
	context_y_arr = Array(reshape(Float32.(collect(context_y)), 1, :, 1))
	target_x_arr = Array(reshape(Float32.(collect(target_x)), 1, :, 1))
	context_x_np = np.asarray(context_x_arr, dtype=np.float32)
	context_y_np = np.asarray(context_y_arr, dtype=np.float32)
	target_x_np = np.asarray(target_x_arr, dtype=np.float32)
	context_x_tensor = torch.from_numpy(context_x_np).to(model.device)
	context_y_tensor = torch.from_numpy(context_y_np).to(model.device)
	target_x_tensor = torch.from_numpy(target_x_np).to(model.device)
	
	# Make predictions with no gradient
	with_no_grad = torch.no_grad()
	with_no_grad.__enter__()
	
	try
		pred_mean, pred_std = model.model(context_x_tensor, context_y_tensor, target_x_tensor)
		
		# Convert back to Julia arrays
		mean = pyconvert(Vector{Float64}, pred_mean[0, pybuiltins.slice(nothing), 0].cpu().numpy())
		std = pyconvert(Vector{Float64}, pred_std[0, pybuiltins.slice(nothing), 0].cpu().numpy())
		
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
- `Vector{Float64}`: Loss values over training iterations
"""
function train_model!(model::TNPModel, sample_fn::Union{Py, Function};
			model_path::Union{Nothing, String} = nothing,
			save_path::Union{Nothing, String} = "data/tnp_model.pt",
			num_iterations::Int = 10_000,
			lr_start::Float64 = 1e-4,
			lr_end::Float64 = 0.0,
			print_freq::Int = 500,
			device::Union{Nothing, String} = nothing)
	
	# Import training function
	train_module = pyimport("train")
	train_tnp = train_module.train_tnp

	py_sample_fn = _to_py_callable(sample_fn)

	# Train model
	losses = train_tnp(
		model.model,
		py_sample_fn,
		num_iterations,
		lr_start,
		lr_end,
		print_freq,
		device,
		model_path,
		save_path
	)

	return pyconvert(Vector{Float64}, losses)
end

_to_py_callable(sample_fn::Py) = sample_fn
_to_py_callable(sample_fn::Function) = Py(sample_fn)
