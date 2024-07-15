using SparseArrays


Base.@kwdef mutable struct LinearProgram
    nRows::Integer = 0
    nCols::Integer = 0
    cost::Vector{Float64} = Vector{Float64}(undef, 0)
    A::SparseMatrixCSC{Float64,Int32} = SparseMatrixCSC{Float64,Int32}(undef, 0, 0)
    lhs::Vector{Float64} = Vector{Float64}(undef, 0)
    rhs::Vector{Float64} = Vector{Float64}(undef, 0)
    l::Vector{Float64} = Vector{Float64}(undef, 0)
    u::Vector{Float64} = Vector{Float64}(undef, 0)
    sense::Int32 = 1
    offset::Float64 = 0.0
end

Base.@kwdef mutable struct Solution
    col_value::Vector{Float64} = Vector{Float64}(undef, 0)
    col_dual::Vector{Float64} = Vector{Float64}(undef, 0)
    row_value::Vector{Float64} = Vector{Float64}(undef, 0)
    row_dual::Vector{Float64} = Vector{Float64}(undef, 0)
    value_valid::Int32 = 0
    dual_valid::Int32 = 0
end

Base.@kwdef mutable struct Information
    status::Int32 = -1
    primal_objective::Float64 = Inf
    dual_objective::Float64 = -Inf
    duality_gap::Float64 = -Inf
    complementarity::Float64 = -Inf
    primal_feasibility::Float64 = Inf
    dual_feasibility::Float64 = Inf
    primal_objective_avg::Float64 = Inf
    dual_objective_avg::Float64 = -Inf
    duality_gap_avg::Float64 = -Inf
    complementarity_avg::Float64 = -Inf
    primal_feasibility_avg::Float64 = Inf
    dual_feasibility_avg::Float64 = Inf
    niter::Int32 = 0
    runtime::Float64 = 0.0
    presolve_time::Float64 = 0.0
    scaling_time::Float64 = 0.0
end


Base.@kwdef mutable struct Parameter
    ifChangeIntParam::Vector{UInt8} = zeros(Int32, Int(N_INT_USER_PARAM) - 1)
    intParam::Vector{Int32} = zeros(Int32, Int(N_INT_USER_PARAM) - 1)
    ifChangeFloatParam::Vector{UInt8} = zeros(Int32, Int(N_FLOAT_USER_PARAM) - 1)
    floatParam::Vector{Float64} = zeros(Float64, Int(N_FLOAT_USER_PARAM) - 1)
end

Base.@kwdef mutable struct cuPDLP_C
    lp::LinearProgram = LinearProgram()
    param::Parameter = Parameter()
    sol::Solution = Solution()
    info::Information = Information()
end

function load_lp!(solver::cuPDLP_C, cost::Vector{Float64}, A::SparseMatrixCSC, lhs::Vector{Float64}, rhs::Vector{Float64}, l::Vector{Float64}, u::Vector{Float64}, sense::Integer, offset::Float64)
    load_lp!(solver.lp, cost, A, lhs, rhs, l, u, sense, offset)
end

function load_lp!(lp::LinearProgram, cost::Vector{Float64}, A::SparseMatrixCSC, lhs::Vector{Float64}, rhs::Vector{Float64}, l::Vector{Float64}, u::Vector{Float64}, sense::Integer, offset::Float64)
    lp.nRows, lp.nCols = size(A)
    lp.cost = cost
    lp.A = SparseMatrixCSC{Float64,Int32}(A)
    lp.lhs = lhs
    lp.rhs = rhs
    lp.l = l
    lp.u = u
    lp.sense = Int32(sense)
    lp.offset = offset

end


function setParam!(solver::cuPDLP_C, param_name, param_value)
    solver.param.setParam!(param_name, param_value)
end

function setParam!(param::Parameter, param_name, param_value)
    println("Setting Parameter [$(param_name)] to $(param_value).")
    if param_name == "nIterLim"
        idx = Int32(N_ITER_LIM)
        param.ifChangeIntParam[idx] = 1
        param.intParam[idx] = Int32(param_value)
    elseif param_name == "ifScaling"
        idx = Int32(IF_SCALING)
        param.ifChangeIntParam[idx] = 1
        param.intParam[idx] = Int32(param_value)
    elseif param_name == "iScalingMethod"
        idx = Int32(I_SCALING_METHOD)
        param.ifChangeIntParam[idx] = 1
        param.intParam[idx] = Int32(param_value)
    elseif param_name == "eLineSearchMethod"
        idx = Int32(E_LINE_SEARCH_METHOD)
        param.ifChangeIntParam[idx] = 1
        param.intParam[idx] = Int32(param_value)
    elseif param_name == "dScalingLimit"
        idx = Int32(D_SCALING_LIMIT)
        param.ifChangeFloatParam[idx] = 1
        param.floatParam[idx] = Float64(param_value)
    elseif param_name == "dPrimalTol"
        idx = Int32(D_PRIMAL_TOL)
        param.ifChangeFloatParam[idx] = 1
        param.floatParam[idx] = Float64(param_value)
    elseif param_name == "dDualTol"
        idx = Int32(D_DUAL_TOL)
        param.ifChangeFloatParam[idx] = 1
        param.floatParam[idx] = param_value
    elseif param_name == "dGapTol"
        idx = Int32(D_GAP_TOL)
        param.ifChangeFloatParam[idx] = 1
        param.floatParam[idx] = param_value
    elseif param_name == "dFeasTol"
        idx = Int32(D_FEAS_TOL)
        param.ifChangeFloatParam[idx] = 1
        param.floatParam[idx] = param_value
    elseif param_name == "dTimeLim"
        idx = Int32(D_TIME_LIM)
        param.ifChangeFloatParam[idx] = 1
        param.floatParam[idx] = param_value
    elseif param_name == "eRestartMethod"
        idx = Int32(E_RESTART_METHOD)
        param.ifChangeIntParam[idx] = 1
        param.intParam[idx] = param_value
    elseif param_name == "ifRuizScaling"
        idx = Int32(IF_RUIZ_SCALING)
        param.ifChangeIntParam[idx] = 1
        param.intParam[idx] = param_value
    elseif param_name == "ifL2Scaling"
        idx = Int32(IF_L2_SCALING)
        param.ifChangeIntParam[idx] = 1
        param.intParam[idx] = param_value
    elseif param_name == "ifPcScaling"
        idx = Int32(IF_PC_SCALING)
        param.ifChangeIntParam[idx] = 1
        param.intParam[idx] = param_value
    elseif param_name == "nLogInt"
        idx = Int32(N_LOG_INTERVAL)
        param.ifChangeIntParam[idx] = 1
        param.intParam[idx] = param_value
    elseif param_name == "ifPre"
        idx = Int32(IF_PRESOLVE)
        param.ifChangeIntParam[idx] = 1
        param.intParam[idx] = param_value
    else
        println("Invalid Parameter Name $(param_name).")
    end
end

function help(solver::cuPDLP_C)
    ccall((:PDHG_PrintUserParamHelper, LIBCUPDLP_PATH * "libcupdlp" * LIBSUFFIX), Cvoid, ())
end


# function solve!(solver::cuPDLP_C)
#     ccall(
#         (:cupdlp_solve, LIBCUPDLP_PATH * "libjlcupdlp" * LIBSUFFIX),
#         Cint,
#         (
#             Cint, Cint, Ptr{Cdouble},
#             Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble},
#             Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
#             Ptr{Cdouble}, Cint, Cdouble,
#             Ptr{Cint}, Ptr{Cint},
#             Ptr{Cint}, Ptr{Cdouble},
#             Ptr{Cint}, Ptr{Cint}, Ptr{Cint},
#             Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
#             Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
#             Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
#             Ptr{Cdouble}, Ptr{Cdouble},
#             Ptr{Cdouble}, Ptr{Cdouble},
#             Ptr{Cdouble}, Ptr{Cdouble},
#             Ptr{Cdouble}, Ptr{Cint}, Ptr{Cdouble},
#             Ptr{Cdouble}, Ptr{Cdouble}
#         ),
#         solver.lp.nRows, solver.lp.nCols, solver.lp.cost,
#         solver.lp.A.colptr, solver.lp.A.rowval, solver.lp.A.nzval,
#         solver.lp.lhs, solver.lp.rhs, solver.lp.l,
#         solver.lp.u, solver.lp.sense, solver.lp.offset,
#         solver.param.ifChangeIntParam, solver.param.intParam,
#         solver.param.ifChangeFloatParam, solver.param.floatParam,
#         solver.info.status, solver.sol.value_valid, solver.sol.dual_valid,
#         solver.sol.col_value, solver.sol.col_dual, solver.sol.row_value,
#         solver.sol.row_dual, solver.info.primal_objective, solver.info.dual_objective,
#         solver.info.duality_gap, solver.info.complementarity, solver.info.primal_feasibility,
#         solver.info.dual_feasibility, solver.info.primal_objective_avg,
#         solver.info.dual_objective_avg, solver.info.duality_gap_avg,
#         solver.info.complementarity_avg, solver.info.primal_feasibility_avg,
#         solver.info.dual_feasibility_avg, solver.info.niter, solver.info.runtime,
#         solver.info.presolve_time, solver.info.scaling_time
#     )
# end

function solve!(solver::cuPDLP_C)
    solver.sol.col_value = Vector{Float64}(undef, solver.lp.nCols)
    solver.sol.col_dual = Vector{Float64}(undef, solver.lp.nCols)
    solver.sol.row_value = Vector{Float64}(undef, solver.lp.nRows)
    solver.sol.row_dual = Vector{Float64}(undef, solver.lp.nRows)

    # 定义 `Ref` 类型的变量
    status = Ref{Int32}()
    value_valid = Ref{Int32}()
    dual_valid = Ref{Int32}()
    primal_obj = Ref{Float64}()
    dual_obj = Ref{Float64}()
    duality_gap = Ref{Float64}()
    comp = Ref{Float64}()
    primal_feas = Ref{Float64}()
    dual_feas = Ref{Float64}()
    primal_obj_avg = Ref{Float64}()
    dual_obj_avg = Ref{Float64}()
    duality_gap_avg = Ref{Float64}()
    comp_avg = Ref{Float64}()
    primal_feas_avg = Ref{Float64}()
    dual_feas_avg = Ref{Float64}()
    niter = Ref{Int32}()
    runtime = Ref{Float64}()
    presolve_time = Ref{Float64}()
    scaling_time = Ref{Float64}()

    colptr = solver.lp.A.colptr .- Int32(1)
    rowval = solver.lp.A.rowval .- Int32(1)

    # 调用 C 函数
    retcode = ccall(
        (:cupdlp_solve, LIBCUPDLP_PATH * "libjlcupdlp" * LIBSUFFIX),  # C 函数名和库路径
        Cint,  # 返回类型
        (
            Cint, Cint,  # nRows, nCols
            Ptr{Cdouble},  # cost
            Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble},  # A_csc_beg, A_csc_idx, A_csc_val
            Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},  # lhs, rhs, lower, upper
            Cint, Cdouble,  # sense, offset
            Ptr{Cuchar}, Ptr{Cint}, Ptr{Cuchar}, Ptr{Cdouble},  # ifChangeIntParam, intParam, ifChangeFloatParam, floatParam
            Ptr{Int32}, Ptr{Int32}, Ptr{Int32},  # status_pdlp, value_valid, dual_valid
            Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},  # col_value, col_dual, row_value, row_dual
            Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},  # primal_obj, dual_obj, duality_gap, comp
            Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},  # primal_feas, dual_feas, primal_obj_avg, dual_obj_avg
            Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},  # duality_gap_avg, comp_avg, primal_feas_avg, dual_feas_avg
            Ptr{Int32}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}  # niter, runtime, presolve_time, scaling_time
        ),
        solver.lp.nRows, solver.lp.nCols,
        solver.lp.cost,
        colptr, rowval, solver.lp.A.nzval,
        solver.lp.lhs, solver.lp.rhs, solver.lp.l, solver.lp.u,
        solver.lp.sense, solver.lp.offset,
        solver.param.ifChangeIntParam, solver.param.intParam, solver.param.ifChangeFloatParam, solver.param.floatParam,
        status, value_valid, dual_valid,
        solver.sol.col_value, solver.sol.col_dual, solver.sol.row_value, solver.sol.row_dual,
        primal_obj, dual_obj, duality_gap, comp,
        primal_feas, dual_feas, primal_obj_avg, dual_obj_avg,
        duality_gap_avg, comp_avg, primal_feas_avg, dual_feas_avg,
        niter, runtime, presolve_time, scaling_time
    )

    # 更新 solver 对象中的信息
    solver.info.status = status[]
    solver.sol.value_valid = value_valid[]
    solver.sol.dual_valid = dual_valid[]
    solver.info.primal_objective = primal_obj[]
    solver.info.dual_objective = dual_obj[]
    solver.info.duality_gap = duality_gap[]
    solver.info.complementarity = comp[]
    solver.info.primal_feasibility = primal_feas[]
    solver.info.dual_feasibility = dual_feas[]
    solver.info.primal_objective_avg = primal_obj_avg[]
    solver.info.dual_objective_avg = dual_obj_avg[]
    solver.info.duality_gap_avg = duality_gap_avg[]
    solver.info.complementarity_avg = comp_avg[]
    solver.info.primal_feasibility_avg = primal_feas_avg[]
    solver.info.dual_feasibility_avg = dual_feas_avg[]
    solver.info.niter = niter[]
    solver.info.runtime = runtime[]
    solver.info.presolve_time = presolve_time[]
    solver.info.scaling_time = scaling_time[]

    nothing
end


Base.getproperty(solver::cuPDLP_C, name::Symbol) =
    name == :load_lp! ? (cost, A, lhs, rhs, l, u, sense, offset) -> load_lp!(solver, cost, A, lhs, rhs, l, u, sense, offset) :
    name == :setParam! ? (key, value) -> setParam!(solver, key, value) :
    name == :help ? () -> help(solver) :
    name == :solve! ? () -> solve!(solver) :
    getfield(solver, name)

Base.getproperty(lp::LinearProgram, name::Symbol) =
    name == :load_lp! ? (cost, A, lhs, rhs, l, u, sense, offset) -> load_lp!(lp, cost, A, lhs, rhs, l, u, sense, offset) :
    getfield(lp, name)

Base.getproperty(param::Parameter, name::Symbol) =
    name == :setParam! ? (key, value) -> setParam!(param, key, value) :
    getfield(param, name)
