using JuMP
using SCIP
using Plots
using Random

const CLAMP_LB = -0.5
const CLAMP_UB = 0.5
const BIG_M = 100
const TOL = 1e-4
const VARIANCE = 0.1

function makedata(n::Int, m::Int)
    a = randn(n, m)
    x = randexp(m)
    b = a * x + VARIANCE * randn(n)
    clamp!(b, CLAMP_LB, CLAMP_UB)
    return a, b, x
end

function solveregression(a::Matrix{Float64}, b::Vector{Float64}, use_clamping::Bool)
    n, m = size(a)
    @assert (n,) == size(b)

    model = Model(SCIP.Optimizer)
    # x needs an lb and ub for the indicator constraints to work.
    # As a heuristic, set these to large values derived from input data.
    maxabsx = BIG_M * maximum(abs, a)
    @variable(model, -maxabsx ≤ x[1:m] ≤ maxabsx)

    @expression(model, bpred_linear, a * x)

    bpred = if use_clamping
        # bpred_clamp = clamp.(bpred_linear, CLAMP_LB, CLAMP_UB)
        @variable(model, clamphelper_lb[1:n], Bin)
        @variable(model, clamphelper_ub[1:n], Bin)
        @constraint(model, [i in 1:n], clamphelper_lb[i] => {bpred_linear[i] ≤ CLAMP_LB})
        @constraint(model, [i in 1:n], !clamphelper_lb[i] => {bpred_linear[i] ≥ CLAMP_LB})
        @constraint(model, [i in 1:n], clamphelper_ub[i] => {bpred_linear[i] ≥ CLAMP_UB})
        @constraint(model, [i in 1:n], !clamphelper_ub[i] => {bpred_linear[i] ≤ CLAMP_UB})
        @expression(model, bpred_clamp[i in 1:n],
            # Start with the linear fit
            bpred_linear[i]
            # If linear fit exceeded the LB (helper == 1) then just
            # return the LB itself, and subtract out the linear fit
            + clamphelper_lb[i] * (CLAMP_LB - bpred_linear[i])
            # If linear fit exceeded the UB (helper == 1) then just
            # return the UB itself, and subtract out the linear fit
            + clamphelper_ub[i] * (CLAMP_UB - bpred_linear[i])
        )

        bpred_clamp
    else
        bpred_linear
    end

    # 1-norm loss
    @variable(model, margin[1:n])
    @constraint(model, bpred .- b .≤ margin)
    @constraint(model, bpred .- b .≥ -margin)
    @objective(model, Min, sum(margin))

    optimize!(model)

    if use_clamping
        # Validate that clamping math worked as expected
        for z in bpred
            @assert CLAMP_LB - TOL ≤ value(z) ≤ CLAMP_UB + TOL
        end
    end
    return (x=value.(x), bpred=value.(bpred))
end

function main()
    a, b, _ = makedata(100, 1)
    res_linear = solveregression(a, b, false)
    res_clamp = solveregression(a, b, true)

    pl = plot()
    scatter!(pl, a[:], b, label="data")
    plot!(pl, z -> z * res_linear.x[1], label="linear fit")
    plot!(pl, z -> clamp(z * res_clamp.x[1], CLAMP_LB, CLAMP_UB), label="clamped fit")
    pl
end

main()
