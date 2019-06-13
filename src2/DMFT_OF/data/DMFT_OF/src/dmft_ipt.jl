module imp_solver
    using Distributed
    using LinearAlgebra
    using DSP
    using PyCall
    using QuadGK
    using Dierckx

    include("./util.jl")
    include("./initialize.jl")
    using .util
    using .initialize

    export ipt_selfcons,hartreefock

    np = pyimport("numpy")
    ss = pyimport("scipy.signal")

    zeta(ω,params) = ω + im * params["infinitesimal"]

    function fermi(ωx::Float64,β)
        return 1/(exp(ωx*β)+1)
    end

    function ipt_Σ2_solver_afm(Aw, nf, U)
        AA  = util.conv_same(Aw[:,1] .* nf,Aw[:,2] .* nf)
        AAB = util.conv_same(AA,Aw[:,2] .* (1. .- nf))

        BB  = util.conv_same(Aw[:,1] .* (1. .- nf),Aw[:,2] .* (1. .- nf))
        BBA = util.conv_same(BB,Aw[:,2] .* nf)

        return -π .* U^2 .* (AAB + BBA)
    end

    function ipt_Σ2_solver_pm(Aw, nf, U)
        Ap = Aw[:,1] .* nf
        App = util.conv_same(Ap,Ap)
        Appp = util.conv_same(Ap, App)
        return -π .* U^2 .* (Appp + Appp[end:-1:1])
    end

    function ipt_selfcons(ω,gloc,dos,params)
        g0 = zeros(ComplexF64,np.shape(gloc))
        isi = zeros(Float64,np.shape(gloc))
        hsi = zeros(Float64,np.shape(gloc))
        A0 = zeros(Float64,np.shape(gloc))
        Σ2 = zeros(ComplexF64,np.shape(gloc))

        magnet = 0.0

        dω = ω[2] - ω[1]

        nf = fermi.(ω,params["β"])

        U = params["local_interaction"]
        t = params["hopping"]
        itermax = params["ipt_itermax"]
        nω = params["nω"]
        η = params["infinitesimal"]
        α = params["mixing"]

        ww = util.simpson(params,nω)

        #minimal cost integration
        nw = floor(Int,nω/4)
        we = util.simpson(params,nw)
        ρe = dos[1:4:nω]
        w  = ω[1:4:nω]

        #guess occupation
        if params["antiferromagnetic"] == true
            Σ1 = U .* [0.0 -1.0]
        else
            Σ1 = U .* [-0.5 -0.5]
        end        
        
        #if params["antiferromagnetic"] == true
            for i = 1:nω
                dum = 0.0
                ζ_up = zeta(ω[i] - Σ1[1],params)
                ζ_down = zeta(ω[i] - Σ1[2],params)
                for e = 1:nw
                    dum += we[e].*ρe[e] / (ζ_up*ζ_down - w[e]^2)
                end
                gloc[i,1] = dum * ζ_down
                gloc[i,2] = dum * ζ_up
            end
        #else
        #    for i = 1:nω
        #        dum = 0.0
        #        ζ = zeta(ω[i],params)
        #        for e = 1:nw
        #            dum += we[e].*ρe[e] / (ζ - w[e])
        #        end
        #        gloc[i,1] = dum
        #    end
        #end
        
        for iter = 1:itermax

            gloc_old = deepcopy(gloc)

            ncalc = zeros(Float64,length(gloc[1,:]))
            for w in 1:nω
                for i in 1:length(gloc[1,:])
                    ncalc[i] += -1/π .* imag(gloc_old[w,i]) .* fermi(ω[w],params["β"]) .* ww[w]
                end
            end
            #if params["antiferromagnetic"] == true                
                Σ1[1] = U .* (ncalc[2] - sum(ncalc)/2)
                Σ1[2] = U .* (ncalc[1] - sum(ncalc)/2)
                magnet = (ncalc[2] - ncalc[1]) / sum(ncalc)
            #end

            #if params["antiferromagnetic"] == true
                g0[:,1] = 1 ./ (ω .+ im*η - t^2 .* gloc_old[:,2])
                g0[:,2] = 1 ./ (ω .+ im*η - t^2 .* gloc_old[:,1])
            #else
            #    g0[:,1] = 1 ./ (ω .+ im*η - t^2 .* gloc_old[:,1])
            #end
            
            for i = 1:length(gloc[1,:])
                A0[:,i] = -imag(g0[:,i]) ./ π
            end

            for i = 1:length(gloc[1,:])
                #if params["antiferromagnetic"] == true
                    isi[:,i] = ipt_Σ2_solver_afm(A0,nf,U) * dω * dω
                #else
                #    isi[:,i] = ipt_Σ2_solver_pm(A0,nf,U) * dω * dω
                #end
                isi[:,i] = 0.5 .* (isi[:,i] + isi[end:-1:1,i])
                hsi[:,i] = -imag.(Util.hilbert(isi[:,i]))
            end

            Σ2 = hsi .+ isi.*im

            #if params["antiferromagnetic"] == true
                for i = 1:nω
                    dum = 0.0
                    ζ_up = zeta(ω[i] .- Σ2[i,1] - Σ1[1],params)
                    ζ_down = zeta(ω[i] .- Σ2[i,2] - Σ1[2],params)
                    for e = 1:nw
                        dum += we[e].*ρe[e] / (ζ_up*ζ_down - w[e]^2)
                    end
                    gloc[i,1] = dum * ζ_down
                    gloc[i,2] = dum * ζ_up
                end
            #else
            #    for i = 1:nω
            #        dum = 0.0
            #        ζ = zeta(ω[i] - Σ2[i,1],params)
            #        for e = 1:nw
            #            dum += we[e].*ρe[e] / (ζ - w[e])
            #        end
            #        gloc[i,1] = dum
            #    end
            #end

            convg, error = util.convergent(gloc_old,gloc,params)
            # if iter % 10 == 0
            #     println("====================================")
            #     println("For Iteration = ", iter)
            #     if params["antiferromagnetic"] == true
            #         println("magnetization = ", magnet)
            #     end
            #     println("total occupation = ", sum(ncalc))
            #     println("error = ",error)
            #     println("====================================")
            # end

            if convg == false
                gloc = util.mixing(gloc_old,gloc,α)
            elseif iter == itermax
                println("Convergent is not achieved, here last result:")
                println("====================================")
                println("For Iteration = ", iter)
                if params["antiferromagnetic"] == true
                    println("magnetization = ", magnet)
                end
                println("total occupation = ", sum(ncalc))
                println("error = ",error)
                println("====================================")
                break
            elseif convg == true
                println("Convergent is achieved")
                println("====================================")
                println("For Iteration = ", iter)
                if params["antiferromagnetic"] == true
                    println("magnetization = ", magnet)
                end
                println("total occupation = ", sum(ncalc))
                println("error = ",error)
                println("====================================")
                break
            end
        end #end selfcons

        return gloc,Σ1,Σ2,magnet

    end

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

    function hartreefock(ω,gloc,dos,params)
        g0 = zeros(ComplexF64,np.shape(gloc))
        isi = zeros(Float64,np.shape(gloc))
        hsi = zeros(Float64,np.shape(gloc))
        A0 = zeros(Float64,np.shape(gloc))
        Σ2 = zeros(ComplexF64,np.shape(gloc))

        #for paramagnetic
        magnet = 0.0

        nf = fermi.(ω,params["β"])

        U = params["local_interaction"]
        t = params["hopping"]
        itermax = params["ipt_itermax"]
        nω = params["nω"]
        η = params["infinitesimal"]
        α = params["mixing"]

        ww = util.simpson(params,nω)

        #minimal cost integration
        nw = floor(Int,nω/8)
        we = util.simpson(params,nw)
        ρe = dos[1:8:nω]
        w  = ω[1:8:nω]

        #guess occupation
        if params["antiferromagnetic"] == true
            Σ1 = U .* [0.0 -1.0]
        else
            Σ1 = U .* [-0.5 -0.5]
        end

        #calculate gloc from Σ_guess        
        if params["antiferromagnetic"] == true
            for i = 1:nω
                dum = 0.0
                dam = 0.0
                ζ_up = zeta(ω[i] - Σ1[1],params)
                ζ_down = zeta(ω[i] - Σ1[2],params)
                for e = 1:nw
                    dum += we[e].*ρe[e] / (ζ_up - w[e])
                    dam += we[e].*ρe[e] / (ζ_down - w[e])
                end
                gloc[i,1] = dum
                gloc[i,2] = dam
            end
        else
            for i = 1:nω
                dum = 0.0
                ζ = zeta(ω[i],params)
                for e = 1:nw
                    dum += we[e].*ρe[e] / (ζ - w[e])
                end
                gloc[i,1] = dum
            end
        end
        
        for iter = 1:itermax

            gloc_old = deepcopy(gloc)

            #calculate ncalc for each band (gloc↑ and gloc↓ )
            ncalc = zeros(Float64,length(gloc[1,:]))
            for w in 1:nω
                for i in 1:length(gloc[1,:])
                    ncalc[i] += -1/π .* imag(gloc_old[w,i]) .* fermi(ω[w],params["β"]) .* ww[w]
                end
            end
            #calculate Σcalc with half-filling shift μ = U/2 * (n↑ + n↓ ) 
            if params["antiferromagnetic"] == true                
                Σ1[1] = U .* (ncalc[2] - sum(ncalc)/2)
                Σ1[2] = U .* (ncalc[1] - sum(ncalc)/2)
                magnet = (ncalc[2] - ncalc[1]) / sum(ncalc)
            end

            #calculate gloc from calculated Σcalc
            if params["antiferromagnetic"] == true
                for i = 1:nω
                    dum = 0.0
                    dam = 0.0
                    ζ_up = zeta(ω[i] - Σ1[1],params)
                    ζ_down = zeta(ω[i] - Σ1[2],params)
                    for e = 1:nw
                        dum += we[e].*ρe[e] / (ζ_up - w[e])
                        dam += we[e].*ρe[e] / (ζ_down - w[e])
                    end
                    gloc[i,1] = dum
                    gloc[i,2] = dam
                end
            else
                for i = 1:nω
                    dum = 0.0
                    ζ = zeta(ω[i],params)
                    for e = 1:nw
                        dum += we[e].*ρe[e] / (ζ - w[e])
                    end
                    gloc[i,1] = dum
                end
            end

            #convergent checking
            convg, error = util.convergent(gloc_old,gloc,params)

            #print result every 10 iterations
            if iter % 10 == 0
                println("====================================")
                println("For Iteration = ", iter)
                if params["antiferromagnetic"] == true
                    println("magnetization = ", magnet)
                end
                println("total occupation = ", sum(ncalc))
                println("error = ",error)
                println("====================================")
            end

            #final conditions
            if convg == false
                gloc = util.mixing(gloc_old,gloc,α)
            elseif iter == itermax
                println("Convergent is not achieved, need more tuning on mixing parameter or number of iterations!")
                println("====================================")
                println("For Iteration = ", iter)
                if params["antiferromagnetic"] == true
                    println("magnetization = ", magnet)
                end
                println("total occupation = ", sum(ncalc))
                println("error = ",error)
                println("====================================")
                break
            elseif convg == true
                println("Convergent is achieved")
                println("====================================")
                println("For Iteration = ", iter)
                if params["antiferromagnetic"] == true
                    println("magnetization = ", magnet)
                end
                println("total occupation = ", sum(ncalc))
                println("error = ",error)
                println("====================================")
                break
            end
        end #end selfcons

        return gloc,Σ1,Σ2,magnet


    end


end
