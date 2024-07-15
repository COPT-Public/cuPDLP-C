module cuPDLPc

include("utilities/cuPDLP_CONST.jl")
include("utilities/cuPDLP-julia-c.jl")

export solve!, setParam!, help, load_lp!, cuPDLP_C

end # module cuPDLPc
