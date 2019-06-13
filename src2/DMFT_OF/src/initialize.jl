module initialize

    using PyCall
    using Dierckx
    using LinearAlgebra

    export bare_dos,setup_init

    np = pyimport("numpy")

    function ϵk(t::Float64,kx::Float64,ky::Float64,kz::Float64)
        return -2t .* (cos(kx) + cos(ky) + cos(kz))
    end;

    function G0k(t::Float64,w::Float64,kx::Float64,ky::Float64,kz::Float64)
        zeta = w + 0.05*im
        dum = [zeta ϵk(t,kx,ky,kz); ϵk(t,kx,ky,kz) zeta]
        return inv(dum)
    end;

    function bare_dos(ω,params)
        if params["System"] == "cubic"

            nkpoint = 50
            nwpoint = 35
            dk      = (2π/(nkpoint-1))

            t = params["hopping"]
            nω = params["nω"]
            ωmin = params["ωrange"][1]
            ωmax = params["ωrange"][2]
            
            w = np.linspace(ωmin,ωmax,nwpoint)

            g0 = zeros(ComplexF64,nwpoint,2,2)
            for i in 1:nwpoint
                dum = zeros(ComplexF64,2,2)
                for kx in 1:nkpoint
                    kxp = -π + dk*(kx-1)
                    for ky in 1:nkpoint
                        kyp = -π + dk*(ky-1)
                        for kz in 1:nkpoint
                            kzp = -π + dk*(kz-1)
                            dum += G0k(t,w[i],kxp,kyp,kzp)                
                        end            
                    end        
                end    
                g0[i,:,:] = dum ./ (nkpoint^3)    
            end
            
            trace = zeros(ComplexF64,nwpoint)
            dos = zeros(Float64,nω)
            for i in 1:nwpoint
                trace[i] = tr(g0[i,:,:])
            end
            splimag = Spline1D(w,imag.(trace))
            for i in 1:nω
                dos[i] = -1/π * splimag(ω[i])
            end

            ns = findall(x -> x == maximum(dos),dos)[1]
            ns = floor(Int,ns)
            ns = [(nω-ns) ns]

            for i = minimum(ns):maximum(ns)
                dos[i] = maximum(dos)
            end

        elseif params["System"] == "bethe" 
            t = params["hopping"]
            η = params["infinitesimal"]
            ζ = ω .+ im*η
            sqr = sqrt.(ζ.^2  .- 4t.^2)
            sqr = sign.(imag.(sqr)).*sqr
            sqr = (ζ - sqr) ./ 2t.^2
            dos = - 1/π .* imag(sqr)
            dos = dos .* 2.0            
        else 
            println("Other System is have not been build yet")
            dos = zeros(Float64,nω)
        end

        return dos
    end

    function setup_init(params)
        dim = 2

        ω = np.linspace(params["ωrange"][1],params["ωrange"][2],params["nω"])
        ωn = π .* (2 .* np.arange(params["n_matsubara"]) .+ 1) ./ params["β"]
        gloc = zeros(ComplexF64,params["nω"],dim)
        g_iwn = zeros(ComplexF64,params["n_matsubara"],dim)
        Σ2_iwn = zeros(ComplexF64,params["n_matsubara"],dim)
        return ω,ωn,gloc,g_iwn,Σ2_iwn
    end

end    
