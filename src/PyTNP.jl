module PyTNP

using PythonCall

export TNPModel, load_model, predict, train_model

include("tnp.jl")

end # module PyTNP
