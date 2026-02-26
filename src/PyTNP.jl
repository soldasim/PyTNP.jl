module PyTNP

using PythonCall

export TNPModel
export DefaultMode, KNNMode
export init_model, load_model, save_model, predict, train_model!

include("python.jl")
include("tnp.jl")
include("tnp_knn.jl")

function __init__()
	_setup_python_path!()
end

end # module PyTNP
