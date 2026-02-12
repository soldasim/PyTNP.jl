module PyTNP

using PythonCall

export TNPModel, init_model, load_model, save_model, predict, train_model!
include("tnp.jl")

function __init__()
	_setup_python_path!()
end

end # module PyTNP
