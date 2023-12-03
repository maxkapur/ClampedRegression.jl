using JuMP
using SCIP
using Plots
using Random

const BIG_M = 100
const CLAMP_LB = -0.5
const CLAMP_UB = 0.5
const TOL = 1e-4
const VARIANCE = 0.1


"""
    solveregression(a, b, use_clamping)

Fit a linear model to the observed data. 

`a` is an n-by-m matrix of observed independent variables, and
`b` is an n-vector of observed dependent variables. Returns a
`NamedTuple` `res`, where `res.x` gives the predicted coefficients
and `res.bpred` gives the predicted values of `b`.

- If `use_clamping` is `false`, the model to be fit is `b = a * x`.
- If `use_clamping` is `true`, the model to be fit is 
`b = clamp.(a * x, CLAMP_LB, CLAMP_UB)`.

In either case, the loss function is the sum of absolute deviations 
between `res.bpred` and `b` (that is, 1-norm loss).
"""
function solveregression(a::Matrix{Float64}, b::Vector{Float64}, use_clamping::Bool)
    n, m = size(a)
    @assert (n,) == size(b)

    # Initialize a JuMP model
    model = Model(SCIP.Optimizer)

    # x needs an lb and ub for the indicator constraints to work.
    # As a heuristic, set these to large values derived from input data.
    # In practice, these bounds should never be binding
    maxabsx = BIG_M * maximum(abs, a)
    @variable(model, -maxabsx ≤ x[1:m] ≤ maxabsx)

    # Set bpred to our desired model
    @expression(model, bpred_linear, a * x)
    bpred = if use_clamping
        # Goal: bpred_clamp = clamp.(bpred_linear, CLAMP_LB, CLAMP_UB)

        # Set up indicator variables of whether bpred_linear exceeded
        # clamp bounds
        @variable(model, clamphelper_lb[1:n], Bin)
        @variable(model, clamphelper_ub[1:n], Bin)
        # note: We may not need both directions of the implication here
        @constraint(model, [i in 1:n], clamphelper_lb[i] => {bpred_linear[i] ≤ CLAMP_LB})
        @constraint(model, [i in 1:n], !clamphelper_lb[i] => {bpred_linear[i] ≥ CLAMP_LB})
        @constraint(model, [i in 1:n], clamphelper_ub[i] => {bpred_linear[i] ≥ CLAMP_UB})
        @constraint(model, [i in 1:n], !clamphelper_ub[i] => {bpred_linear[i] ≤ CLAMP_UB})

        # Define bpred_clamp using the indicator variables
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

    # Set up 1-norm loss objective function
    @variable(model, margin[1:n])
    @constraint(model, bpred .- b .≤ margin)
    @constraint(model, bpred .- b .≥ -margin)
    @objective(model, Min, sum(margin))

    # Solve the problem
    optimize!(model)

    # Warn if our heuristic bounds on x turned out to be binding
    if !all(-maxabsx + TOL < value(z) < maxabsx - TOL for z in x)
        @warn "x exceeded heuristic bounds"
    end

    # Validate that clamping math worked as expected
    if use_clamping
        @assert all(CLAMP_LB - TOL ≤ value(z) ≤ CLAMP_UB + TOL for z in bpred)
    end
    return (x=value.(x), bpred=value.(bpred))
end


"""
    makedata(n, m)

Make some fake data with `n` observations and `m` independent variables. 
Return a `NamedTuple` with fields `a`, `b`, and `x`, giving the independent
variables, dependent variables, and ground-truth coefficients, respectively.
"""
function makedata(n::Int, m::Int)
    a = randn(n, m)
    x = randexp(m)
    b = a * x + VARIANCE * randn(n)
    clamp!(b, CLAMP_LB, CLAMP_UB)
    return (; a, b, x)
end


"""
    main()

Solve a sample problem with a single independent variable, with and
without clamping, and generate a plot comparing the results.
"""
function main()
    data = makedata(100, 1)
    res_linear = solveregression(data.a, data.b, false)
    res_clamp = solveregression(data.a, data.b, true)

    pl = plot()
    scatter!(pl, data.a[:], data.b, label="data")
    plot!(pl, z -> z * res_linear.x[1], label="linear fit")
    plot!(pl, z -> clamp(z * res_clamp.x[1], CLAMP_LB, CLAMP_UB), label="clamped fit")
    pl
end

main()
