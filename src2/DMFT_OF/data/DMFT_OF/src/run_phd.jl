using Statistics
using DSP
using QuadGK
using SharedArrays
using JLD2

@everywhere include("./util.jl")
@everywhere include("./initialize.jl")
@everywhere include("./dmft_ipt.jl")
using .util
using .initialize
using .imp_solver

println("This is magnetic phase diagram within IPT solver")
println("By Muhammad Gaffar")
println("")

println("number of processors used = ", nworkers())
println("")

params = Dict(  "System"            => "cubic",
                "antiferromagnetic" => true,
                "hopping"           => + 0.500,
                "local_interaction" => + 2.000,
                "β"                 => + 3.00,
                "n_matsubara"       => + 2^10,
                "nω"                => + 2^11,
                "n_fluctuation"     => + 30,
                "ωrange"            => + [-16.0 16.0],
                "infinitesimal"     => + 0.05,
                "ipt_itermax"       => + 200,
                "of_itermax"        => + 15,
                "tolerance"         => + 1e-2,
                "mixing"            => + 0.40);


println("data initialization started...")
ω,ωn,gloc,g_iωn,Σ2_iωn = initialize.setup_init(params);
bare_dos = initialize.bare_dos(ω,params);
println("data initialization finished...")
println("")


println("IPT Phase Diagram is being Calculated, it will takes long time...")

T = range(0.01,length = 20, stop = 3.01)
U = range(0.0,length = 10, stop = 8.0)

gloc_IPTPH = SharedArray{ComplexF64}(length(T),length(U),params["nω"],2);
Σ1_IPTPH = SharedArray{Float64}(length(T),length(U),2);
Σ2_IPTPH = SharedArray{ComplexF64}(length(T),length(U),params["nω"],2);
magnet_IPTPH = SharedArray{Float64}(length(T),length(U));

@time @inbounds @sync @distributed for iT in 1:length(T)
    params["β"] = 1 ./ T[iT]
    @inbounds for iU in 1:length(U)
        params["local_interaction"] = U[iU]
	println("for iT in $iT and iU in $iU")
        gloc_IPTPH[iT,iU,:,:],
        Σ1_IPTPH[iT,iU,:],
        Σ2_IPTPH[iT,iU,:,:],
        magnet_IPTPH[iT,iU] = imp_solver.ipt_selfcons(ω,gloc,bare_dos,params);
    end
end

println("IPT Calculation is Done...")
println("")

println("OF Phase Diagram is being Calculated, it will takes long time...")

gloc_OFPH = SharedArray{ComplexF64}(length(T),length(U),params["nω"],2);
Σ2ωn_OFPH = SharedArray{ComplexF64}(length(T),length(U),params["n_matsubara"],2);
Σ2_OFPH = SharedArray{ComplexF64}(length(T),length(U),params["nω"],2);
magnet_OFPH = SharedArray{Float64}(length(T),length(U));
probs = SharedArray{Float64}(length(T),length(U),params["n_fluctuation"],params["n_fluctuation"]);

@time @inbounds @sync @distributed for iT in 1:length(T)
    params["β"] = 1 ./ T[iT]
    @inbounds for iU in 1:length(U)
        println("for iT in $iT and iU in $iU")
        params["local_interaction"] = U[iU]
        gloc_OFPH[iT,iU,:,:],
        Σ2ωn_OFPH[iT,iU,:,:],
        Σ2_OFPH[iT,iU,:,:],
        magnet_OFPH[iT,iU],
        probs[iT,iU,:,:] = imp_solver.OF_solver(ω,bare_dos,gloc_IPTPH[iT,iU,:,:],Σ1_IPTPH[iT,iU,:],Σ2_IPTPH[iT,iU,:,:],params);
    end
end

println("OF Calculation is Done...")
println("")

println("Now make data file...")
gloc_IPTPH = convert(Array{ComplexF64,4},gloc_IPTPH)
Σ1_IPTPH = convert(Array{Float64,3},Σ1_IPTPH)
Σ2_IPTPH = convert(Array{ComplexF64,4},Σ2_IPTPH)
magnet_IPTPH = convert(Array{Float64,2},magnet_IPTPH);

gloc_OFPH = convert(Array{ComplexF64,4},gloc_OFPH)
Σ2ωn_OFPH = convert(Array{ComplexF64,4},Σ2ωn_OFPH)
Σ2_OFPH = convert(Array{ComplexF64,4},Σ2_OFPH)
magnet_OFPH = convert(Array{Float64,2},magnet_OFPH)
probs = convert(Array{Float64,4},probs);

@save "common.jld2" ω bare_dos U T
@save "ph_diagram_ipt.jld2" gloc_IPTPH Σ1_IPTPH Σ2_IPTPH magnet_IPTPH 
@save "ph_diagram_of.jld2" gloc_OFPH Σ2ωn_OFPH Σ2_OFPH magnet_OFPH probs
println("data file is done...")
