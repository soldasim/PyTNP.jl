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
	load_model(model_path::String = "tnp_model.pt";
			   x_dim::Int = 1,
			   y_dim::Int = 1,
			   dim_model::Int = 128,
			   num_heads::Int = 4,
			   num_encoder_layers::Int = 2,
			   dim_feedforward::Int = 512,
			   dropout::Float64 = 0.1)

Load the trained TNP model from a file.

# Arguments
- `model_path::String`: Path to the saved model weights
- Model architecture parameters (must match training configuration)

# Example
```julia
model = PyTNP.load_model("tnp_model.pt")
```
"""
function load_model(model_path::String = "tnp_model.pt";
					x_dim::Int = 1,
					y_dim::Int = 1,
					dim_model::Int = 128,
					num_heads::Int = 4,
					num_encoder_layers::Int = 2,
					dim_feedforward::Int = 512,
					dropout::Float64 = 0.1)
	_setup_python_path!()
	
	# Import Python modules
	torch = pyimport("torch")
	tnp_module = pyimport("tnp")
	TransformerNeuralProcess = tnp_module.TransformerNeuralProcess
    
	# Determine device
	device = if pyconvert(Bool, torch.backends.mps.is_available())
		"mps"
	elseif pyconvert(Bool, torch.cuda.is_available())
		"cuda"
	else
		"cpu"
	end
    
	# Initialize model
	model = TransformerNeuralProcess(
		x_dim = x_dim,
		y_dim = y_dim,
		dim_model = dim_model,
		num_heads = num_heads,
		num_encoder_layers = num_encoder_layers,
		dim_feedforward = dim_feedforward,
		dropout = dropout
	)
    
	# Load weights
	state_dict = torch.load(model_path, map_location=device, weights_only=true)
	model.load_state_dict(state_dict)
	model = model.to(device)
	model.eval()

	println("Model loaded successfully on device: $(device)")
	return TNPModel(model, device)
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
model = PyTNP.load_model("tnp_model.pt")

# Define context and target points
context_x = [-1.5, -1.0, -0.5, 0.0, 0.5]
context_y = [1.2, 0.8, 0.3, 0.1, 0.4]
target_x = range(-2, 2, length=50) |> collect

# Get predictions
mean, std = PyTNP.predict(model, context_x, context_y, target_x)

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
	train_model(; model_path::Union{Nothing, String} = nothing,
			save_path::Union{Nothing, String} = "tnp_model.pt",
			x_dim::Int = 1,
			y_dim::Int = 1,
			dim_model::Int = 128,
			num_heads::Int = 4,
			num_encoder_layers::Int = 2,
			dim_feedforward::Int = 512,
			dropout::Float64 = 0.1,
			num_iterations::Int = 10000,
			batch_size::Int = 16,
			num_context_range::Tuple{Int, Int} = (3, 50),
			num_total_points::Int = 100,
			learning_rate::Float64 = 1e-4,
			x_range::Tuple{Float64, Float64} = (-2.0, 2.0),
			kernel_length_scale::Float64 = 0.4,
			kernel_variance::Float64 = 1.0,
			noise_variance::Float64 = 0.01,
			print_freq::Int = 1000,
			device::Union{Nothing, String} = nothing)

Train the Transformer Neural Process model using the Python training loop.

# Arguments
- `model_path`: Optional path to initial weights to load before training
- `save_path`: Optional path to save the trained weights
- Model architecture parameters (must match training configuration)
- Training parameters (see `train.py`)

# Returns
- `Vector{Float64}`: Loss values over training iterations
"""
function train_model(; model_path::Union{Nothing, String} = nothing,
			save_path::Union{Nothing, String} = "data/tnp_model.pt",
			x_dim::Int = 1,
			y_dim::Int = 1,
			dim_model::Int = 128,
			num_heads::Int = 4,
			num_encoder_layers::Int = 2,
			dim_feedforward::Int = 512,
			dropout::Float64 = 0.1,
			num_iterations::Int = 10000,
			batch_size::Int = 16,
			num_context_range::Tuple{Int, Int} = (3, 50),
			num_total_points::Int = 100,
			learning_rate::Float64 = 1e-4,
			x_range::Tuple{Float64, Float64} = (-2.0, 2.0),
			kernel_length_scale::Float64 = 0.4,
			kernel_variance::Float64 = 1.0,
			noise_variance::Float64 = 0.01,
			print_freq::Int = 1000,
			device::Union{Nothing, String} = nothing)
	_setup_python_path!()
	
	# Import training function
	train_module = pyimport("train")
	train_tnp = train_module.train_tnp

	# Train model
	losses = train_tnp(
		nothing,
		x_dim,
		y_dim,
		dim_model,
		num_heads,
		num_encoder_layers,
		dim_feedforward,
		dropout,
		num_iterations,
		batch_size,
		num_context_range,
		num_total_points,
		learning_rate,
		x_range,
		kernel_length_scale,
		kernel_variance,
		noise_variance,
		print_freq,
		device,
		model_path,
		save_path
	)

	return pyconvert(Vector{Float64}, losses)
end
