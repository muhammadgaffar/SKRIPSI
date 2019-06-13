
using PyPlot
using DSP
using QuadGK

include("../src/num.jl")
include("../src/phy.jl")
using .num
using .physics

const t = 0.5
U = 1.6
T = 50.
const nωn = 2^12
const nω = 2^12
const Nd = 25
ωrange = [-4.0,4.0]
const zeroplus = 0.01
const itermax = 200
const tol = 0.001
const mix = 0.50;
kB = 8.617333262145e-5;

ω = range(ωrange[1],length=nω,stop=ωrange[2])
ω = convert(Array{Float64},ω);

β = 1. / kB / T

ωn = π .* (2 .* collect(1:nωn) .+ 1) ./ β;

@time D0ω = baredos.("cubic",t,ω);

#function OF_solver(ω,bare_dos)
    
        Σ1 = U .* [0.0 -0.0]

        ncalc = Σ1 ./ U
        
        df,wd = zeros(Float64,2,Nd),zeros(Float64,Nd)
        
        g_iωn = zeros(ComplexF64,nωn,2)
        Σcalc_iωn = zeros(ComplexF64,nωn,2)
        Σloc_iωn = zeros(Float64,2)
        Σloc = zeros(Float64,2)
        Σ1_f = zeros(Float64,size(Σ1))
        Σ1_fωn = zeros(Float64,size(Σ1))
        Seff = zeros(ComplexF64,Nd,Nd)
        P = zeros(Float64,Nd,Nd)
        Glocs = zeros(ComplexF64,nωn,2,Nd,Nd)
        Glocs_r = zeros(ComplexF64,nω,2,Nd,Nd)
        
        ncalc_f =  zeros(Float64,size(Σ1))
        ncalc_fr =  zeros(Float64,size(Σ1))

        magnet = 0.0
        
        dx,wd = QuadGK.gauss(Nd)
        wdx = wd * transpose(wd)
        for i = 1:2
            df[i,:] = sign(vec([1 -1])[i] ./ U) .* dx
        end

        #minimal cost integration for matsubara
        nw = floor(Int,nωn/4)
        wn  = ωn[1:4:nωn]
        #minimal cost integration for real
        nwr = floor(Int,nω/4)
        ρr = D0ω[1:4:nω]
        wr  = ω[1:4:nω]

        for (iωn,ωnx) in enumerate(ωn)
            ζ_up = im*ωn[iωn] - Σ1[1]
            ζ_down = im*ωn[iωn] - Σ1[2]
        
            intg = ρr ./ (ζ_up*ζ_down .- wr.^2.)
            sum = trapz(wr,intg)        
        
            g_iωn[iωn,1] = sum * ζ_down
            g_iωn[iωn,2] = sum * ζ_up
        end

        @time for m_iter in 1:itermax
            global Σloc_iωn
            global Σcalc_iωn
            global P, Z
            
            giωn_old = deepcopy(g_iωn)
            Σcalciωn_old = deepcopy(Σcalc_iωn)

            #Self energy fluctuation
            Σωn_fluc = Σ1_fωn .+ Σcalc_iωn
    
            gmf_iωn = 1. ./ ( (1. ./ g_iωn) .+ Σωn_fluc)
    
            @fastmath @inbounds for (iNd1,dfn1) in enumerate(df[1,:])
                @fastmath @inbounds for (iNd2,dfn2) in enumerate(df[2,:])

                    Σloc_iωn[1] = U .* (ncalc[1].+ dfn1)
                    Σloc_iωn[2] = U .* (ncalc[2].+ dfn2)
                    Σloc_iωn = reshape(Σloc_iωn,1,2)

                    Glocsinv = (1. ./ gmf_iωn) .- Σloc_iωn

                    determ = gmf_iωn .* Glocsinv
                    determ = determ[:,1] .* determ[:,2]

                    Seff[iNd1,iNd2] = -sum(log.(determ))

                    Glocs[:,:,iNd1,iNd2] = 1 ./ Glocsinv

                end
            end
    
            P = exp.(-real.(Seff))
            Z = sum(P .* wdx)
    
            Gave = zeros(ComplexF64,size(g_iωn))
            @fastmath @inbounds for i = 1:nωn
                @fastmath @inbounds for dim in 1:2 Gave[i,dim] = sum(P[:,:] .* Glocs[i,dim,:,:] .* wdx) end
            end
            Gave = Gave ./ Z
    
            Σcalc_iωn = 1. ./ gmf_iωn .- 1. ./ Gave

            @fastmath @inbounds for (iωn,ωnx) in enumerate(ωn)
                ζup = im*ωn[iωn] - Σcalc_iωn[1]
                ζdw = im*ωn[iωn] - Σcalc_iωn[2]
                
                intg = ρr ./ (ζup*ζdw .- wr.^2.)
                sum = trapz(wr,intg) 
                
                g_iωn[iωn,1] = sum * ζdw
                g_iωn[iωn,2] = sum * ζup
            end
    
            convg, error = convergent(giωn_old,g_iωn,ωn,nωn,tol)
    
            Σcalc_iωn = mixing(Σcalciωn_old,Σcalc_iωn,mix)
    
            println("error = $error, isconvg = $convg")
    
            if convg==true
                break
            end
    
        end

#end

Σ1 = U .* [0.1 -0.1]
ncalc = Σ1 ./ U

g_loc = zeros(ComplexF64,nω,2)
Σcalc = zeros(ComplexF64,nω,2)
Σcalc_r = zeros(ComplexF64,nω,2)

nf = fermi.(ω,T)

for (iω,ωx) in enumerate(ω)
    ζ_up = ω[iω] + im * zeroplus - Σ1[1]
    ζ_down = ωn[iω] + im * zeroplus - Σ1[2]
        
    intg = ρr ./ (ζ_up*ζ_down .- wr.^2.)
    sum = trapz(wr,intg)   
        
    g_loc[iω,1] = sum * ζ_down
    g_loc[iω,2] = sum * ζ_up
end

@time for r_iter = 1:itermax
    global Σcalc_r, ncalc
    global Σloc
    
    gloc_old = deepcopy(g_loc)

    Σ_fluc = Σcalc_r
    gmf = 1. ./ ( (1. ./ g_loc) .+ Σ_fluc );

    for (iNd1,dfn1) in enumerate(df[1,:])
        for (iNd2,dfn2) in enumerate(df[2,:])
            Σloc[1] = U .* (ncalc[1] .+ dfn1)
            Σloc[2] = U .* (ncalc[2] .+ dfn2)
            Σloc = reshape(Σloc,1,2)

            Glocs_r[:,:,iNd1,iNd2] = 1 ./ ( (1. ./ gmf) .- Σloc)
        end
    end

    Gave = zeros(ComplexF64,size(g_loc))
    @fastmath @inbounds for i = 1:nω
        @fastmath @inbounds for dim in 1:2 Gave[i,dim] = sum(P .* Glocs_r[i,dim,:,:] .* wdx) end
    end
    Gave = Gave ./ Z;

    Σcalc_r = 1 ./ gmf .- 1 ./ Gave;

    @fastmath @inbounds for i = 1:nω
        ζup = ω[i] + im * zeroplus - Σcalc_r[i,1]
        ζdw = ω[i] + im * zeroplus - Σcalc_r[i,2]
    
        intg = ρr ./ (ζup*ζdw .- wr.^2)
        sum = trapz(wr,intg)
    
        g_loc[i,1] = sum * ζdw
        g_loc[i,2] = sum * ζup
    end

    ncalc = zeros(Float64,2)
    @fastmath @inbounds for i in 1:2
        ncalc[i] += -1/π .* trapz(ω,imag(g_loc[:,i]) .* nf)
    end
    
    Σcalc_r[:,1] = U * (ncalc[1] - sum(ncalc)) .* ones(Float64,nω)
    Σcalc_r[:,2] = U * (ncalc[2] - sum(ncalc)) .* ones(Float64,nω)
    
    magnet = (ncalc[2] - ncalc[1]) / sum(ncalc)

    convg, error = convergent(gloc_old,g_loc,ω,nω,tol)
    
    println("error = $error, isconvg = $convg")
    
    if convg==true
        break
    end

end #real iteration end        


0.8 -1

plt.plot(ω,-real(Σcalc_r))

function OF_solver(ω,bare_dos,gloc,Σ1,Σ2,params)
    
        Σ1 = reshape(Σ1,1,2)
        
        ncalc = Σ1 ./ params["local_interaction"]
        println("IPT Occupation Calculation = $ncalc")
        #println("")        
        
        #println("Hilbert transformation of gloc and Σ2")
        _,ωn,_,g_iωn,Σ2_iωn = initialize.setup_init(params);
        g_iωn = util.real2matsubara(ωn,g_iωn,ω,gloc,params);
        Σ2_iωn = util.real2matsubara(ωn,Σ2_iωn,ω,Σ2,params);
        #println("")
        
        Nd = params["n_fluctuation"]
        df,wd = zeros(Float64,2,Nd),zeros(Float64,Nd)
        
        Σloc_iωn = zeros(Float64,2)
        Σloc = zeros(Float64,2)
        Σ1_f = zeros(Float64,size(Σ1))
        Σ2_f = zeros(ComplexF64,size(Σ2))
        Σ1_fωn = zeros(Float64,size(Σ1))
        Σ2_fωn = zeros(ComplexF64,size(Σ2_iωn))
        Seff = zeros(ComplexF64,Nd,Nd)
        P = zeros(Float64,Nd,Nd)
        a,b = size(g_iωn)
        Glocs = zeros(ComplexF64,a,b,Nd,Nd)
        a,b = size(gloc)
        Glocs_r = zeros(ComplexF64,a,b,Nd,Nd)
        
        ncalc_f =  zeros(Float64,size(ncalc))
        ncalc_fr =  zeros(Float64,size(ncalc))
        magnet = 0.0
        
        dx,wd = QuadGK.gauss(Nd)
        for i = 1:2
            df[i,:] = sign(ncalc[i]) .* dx
        end

        Σ_ωnipt = Σ2_iωn .+ Σ1 
        Σ_ipt  = Σ2 .+ Σ1

        #minimal cost integration for matsubara
        nωn = params["n_matsubara"]
        nw = floor(Int,nωn/4)
        we = util.simpson(params,nw)
        ρe = bare_dos[1:4:nωn]
        w  = ωn[1:4:nωn]
        #minimal cost integration for real
        nω = params["nω"]
        nwr = floor(Int,nω/4)
        wer = util.simpson(params,nwr)
        ρr = bare_dos[1:4:nω]
        wr  = ω[1:4:nω]
        #simpson weight
        ww = util.simpson(params,nω)

        for m_iter = 1:params["of_itermax"]
            #println("  ")
            #println("--Matsubara Iteration in $m_iter ---------")
            #println("  ")
            
            ncalc_old = deepcopy(ncalc)

            #Self energy fluctuation
            Σωn_fluc = Σ1_fωn .+ Σ2_fωn

            gmf_iωn = 1. ./ ( (1. ./ g_iωn) .+ Σ_ωnipt .+ Σωn_fluc)

            for (iNd1,dfn1) in enumerate(df[1,:])
                for (iNd2,dfn2) in enumerate(df[2,:])

                    Σloc_iωn[1] = params["local_interaction"] .* (ncalc[1].+ dfn1)
                    Σloc_iωn[2] = params["local_interaction"] .* (ncalc[2].+ dfn2)
                    Σloc_iωn = reshape(Σloc_iωn,1,2)

                    Glocsinv = (1. ./ gmf_iωn) .- Σloc_iωn .- Σ2_iωn .- Σ2_fωn

                    determ = gmf_iωn .* Glocsinv
                    determ = determ[:,1] .* determ[:,2]

                    Seff[iNd1,iNd2] = -sum(log.(determ))

                    Glocs[:,:,iNd1,iNd2] = 1 ./ Glocsinv

                end
            end

            Z = 0.0
            for (iNd1,wd1) in enumerate(wd)
                for (iNd2,wd2) in enumerate(wd)

                    @inbounds P[iNd1,iNd2] = exp(-real(Seff[iNd1,iNd2]))

                    @inbounds Z += P[iNd1,iNd2] .* wd1 .* wd2

                end
            end

            Gave = zeros(ComplexF64,size(g_iωn))
            for i = 1:params["n_matsubara"]
                for (iNd1,wd1) in enumerate(wd)
                    for (iNd2,wd2) in enumerate(wd)

                        @inbounds Gave[i,:] += P[iNd1,iNd2] .* Glocs[i,:,iNd1,iNd2] .* wd1 .* wd2

                    end
                end
            end
            Gave = Gave ./ Z

            Σcalc_iωn = 1 ./ gmf_iωn .- 1 ./ Gave

            #retrieve calculated fluctuation in second order self-energy
            Σ2_fi = imag(Σcalc_iωn) - imag(Σ2_iωn)
            Σ2_fωn = 0.0 .+ im .* Σ2_fi;

            for i = 1:nωn
                dum = 0.0
                @inbounds ζup = zeta(im * ωn[i] .- im * params["infinitesimal"] .- Σ_ωnipt[1] .- Σ1_fωn[1] .- Σ2_fωn[i,1],params)
                @inbounds ζdw = zeta(im * ωn[i] .- im * params["infinitesimal"] .- Σ_ωnipt[2] .- Σ1_fωn[2] .- Σ2_fωn[i,2],params)
                for e = 1:nw
                    @inbounds dum += we[e] * ρe[e] / (ζup * ζdw - w[e]^2)
                end
                g_iωn[i,1] = dum * ζdw
                g_iωn[i,2] = dum * ζup
            end

            # -- Real Iteration ----------------

            for r_iter = 1:2

                gmf = 1. ./ ( (1. ./ gloc) .+ Σ_ipt .+ Σ1_f .+ Σ2_f );

                for (iNd1,dfn1) in enumerate(df[1,:])
                    for (iNd2,dfn2) in enumerate(df[2,:])
                            Σloc[1] = params["local_interaction"] .* (ncalc[1] .+ dfn1)
                            Σloc[2] = params["local_interaction"] .* (ncalc[2] .+ dfn2)
                            Σloc = reshape(Σloc,1,2)

                            Glocs_r[:,:,iNd1,iNd2] = 1 ./ ( (1. ./ gmf) .- Σloc .- Σ2 .- Σ2_f)
                    end
                end

                Gave = zeros(ComplexF64,size(gloc))
                for i = 1:params["nω"]
                    for (iNd1,wd1) in enumerate(wd)
                        for (iNd2,wd2) in enumerate(wd)
                                @inbounds Gave[i,:] += P[iNd1,iNd2] .* Glocs_r[i,:,iNd1,iNd2] .* wd1 .* wd2
                        end
                    end
                end
                Gave = Gave ./ Z;

                Σcalc_r = 1 ./ gmf .- 1 ./ Gave

                #retrieve fluctuation for real second order self energy
                Σ2_fi = imag(Σcalc_r) .- imag(Σ2)
                Σ2_fr = zeros(Float64,size(Σ2_fi))
                for i = 1:2
                    Σ2_fr[:,i] = -imag.(Util.hilbert(Σ2_fi[:,i]))
                end
                Σ2_f = Σ2_fr .+ im .* Σ2_fi;
                Σ2_f = 0.05 .* Σ2_f;

                for i = 1:nω
                    dum = 0.0
                    @inbounds ζup = zeta(ω[i] .- Σ1[1] .- Σ1_f[1] .- Σ2[i,1] .- Σ2_f[i,1],params)
                    @inbounds ζdw = zeta(ω[i] .- Σ1[2] .- Σ1_f[2] .- Σ2[i,2] .- Σ2_f[i,2],params)
                    for e = 1:nwr
                        @inbounds dum += wer[e] .* ρr[e] / (ζup * ζdw - wr[e]^2)
                    end
                    gloc[i,1] = dum * ζdw
                    gloc[i,2] = dum * ζup
                end

                ww = util.simpson(params,nω)  
                ncalc = zeros(Float64,2)
                for w in 1:nω
                    for i in 1:2
                        @inbounds ncalc[i] += -1/π .* imag(gloc[w,i]) .* fermi(ω[w],params["β"]) .* ww[w]
                    end
                end

                magnet = (ncalc[2] - ncalc[1]) / sum(ncalc)
                fill = sum(ncalc)

                Σ1_f[1] = params["local_interaction"] .* (ncalc[2] - sum(ncalc)/2) .- Σ1[1]
                Σ1_f[2] = params["local_interaction"] .* (ncalc[1] - sum(ncalc)/2) .- Σ1[2]

                ncalc = ( Σ1_f + Σ1 ) ./ params["local_interaction"]

                # println(" ---Real Iteration in $r_iter--------------------  ")
                # println(" | ncalc = ", ncalc)        
                # println(" | magnet = ", magnet)
                # println(" | total occupation = ", fill)

            end #real iteration end        

            Σ1_fωn[1] = params["local_interaction"] .* ncalc[1] .- Σ1[1]
            Σ1_fωn[2] = params["local_interaction"] .* ncalc[2] .- Σ1[2]
            
            error = abs(ncalc_old[1] .- ncalc[1])
            #println(" error = ", error)
            
            if error <= 6e-4 || m_iter == params["of_itermax"]
                println(" error = $error | done in iteration # $m_iter ")
                break
            end

        end
        
        return gloc, Σ2_iωn .+ Σ2_fωn, Σ2 .+ Σ2_f, magnet, P

    end

