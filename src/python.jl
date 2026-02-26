
function _setup_python_path!()
	sys = pyimport("sys")
	src_path = abspath(@__DIR__)
	if pyconvert(Bool, src_path ∉ sys.path)
		sys.path.insert(0, src_path)
	end
end
