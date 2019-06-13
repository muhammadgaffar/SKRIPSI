
using PyPlot

using Distributed
using SharedArrays

addprocs(4)

@everywhere include("../src/num.jl")
@everywhere include("../src/phy.jl")
using .num
using .physics

@everywhere using DSP

const t = 0.5
U = 3.2
T = 0.0
const nωn = 2^12
const nω = 2^12
ωrange = [-4.0,4.0]
const zeroplus = 0.01
const itermax = 200
const tol = 0.01
const mix = 0.80;

ω = range(ωrange[1],length=nω,stop=ωrange[2])
ω = convert(Array{Float64},ω);

@time D0ω = baredos.("bethe",t,ω);
gloc = zeros(ComplexF64,nω,2);

@everywhere function ipt_solver(Aw, nf, U)
    AA  = num.conv_same(Aw[:,1] .* nf,Aw[:,2] .* nf)
    AAB = num.conv_same(AA,Aw[:,2] .* nf)

    BB  = num.conv_same(Aw[:,1] .* nf,Aw[:,2] .* nf)
    BBA = num.conv_same(BB,Aw[:,2] .* nf)

    return -π .* U^2 .* (AAB + BBA)
end

@everywhere function ipt_selfcons(ω,gloc,dos,t,U,T,itermax,nω,zeroplus,mix,tol)
        g0 = zeros(ComplexF64,size(gloc))
        isi = zeros(Float64,size(gloc))
        hsi = zeros(Float64,size(gloc))
        A0 = zeros(Float64,size(gloc))
        Σ2 = zeros(ComplexF64,size(gloc))

        magnet = 0.0

        dω = ω[2] - ω[1]

        nf = physics.fermi.(ω,T)

        η = zeroplus
        α = mix

        ρe = dos[1:4:nω]
        w  = ω[1:4:nω]

        Σ1 = U .* [0.0 -0.0]
    
        for i = 1:nω
            ζ_up = physics.zeta(ω[i] - Σ1[1],η)
            ζ_down = physics.zeta(ω[i] - Σ1[2],η)
        
            intg = ρe ./ (ζ_up*ζ_down .- w.^2.)
            sum = num.trapz(w,intg)
        
            gloc[i,1] = sum * ζ_down
            gloc[i,2] = sum * ζ_up
        end
        
        for iter = 1:itermax

            gloc_old = deepcopy(gloc)

            ncalc = zeros(Float64,length(gloc[1,:]))
            for i in 1:length(gloc[1,:])
                intg = -1/π .* imag(gloc_old[:,i]) .* nf
                ncalc[i] = num.trapz(ω,intg) 
            end
        
            Σ1[1] = U .* (ncalc[2] - sum(ncalc)/2)
            Σ1[2] = U .* (ncalc[1] - sum(ncalc)/2)
            magnet = (ncalc[2] - ncalc[1]) / sum(ncalc)
        
            g0[:,1] = 1. ./ (ω .+ im*η .- t^2 .* gloc_old[:,2])
            g0[:,2] = 1. ./ (ω .+ im*η .- t^2 .* gloc_old[:,1])
            
            for i = 1:length(gloc[1,:]) A0[:,i] = -imag(g0[:,i]) ./ π end

            
            #isi[:,1] = ipt_solver(A0[:,1],A0[:,2],nf,U) * dω * dω
            #isi[:,2] = ipt_solver(A0[:,1],A0[:,2],nf,U) * dω * dω
            for i = 1:length(gloc[1,:])
                isi[:,i] = ipt_solver(A0,nf,U) * dω * dω
                isi[:,i] = 0.5 .* (isi[:,i] + isi[end:-1:1,i])
                hsi[:,i] = -imag.(Util.hilbert(isi[:,i]))
            end

            Σ2 = hsi .+ im .* isi

            for i = 1:nω
                ζ_up = physics.zeta(ω[i] - Σ1[1] .- Σ2[i,1],η)
                ζ_down = physics.zeta(ω[i] - Σ1[2] .- Σ2[i,2],η)

                intg = ρe ./ (ζ_up*ζ_down .- w.^2.)
                sum = num.trapz(w,intg)

                gloc[i,1] = sum * ζ_down
                gloc[i,2] = sum * ζ_up
            end

            convg, error = num.convergent(gloc_old,gloc,ω,nω,tol)

            if convg == false
                gloc = num.mixing(gloc_old,gloc,mix)
            elseif iter == itermax
                println("Convergent is not achieved. Try Lower Mixings or Higher Iterations")
                break
            elseif convg == true
                println("Convergent is achieved for U = $U, and T = $T K")
                break
            end
        end

        return gloc,Σ1,Σ2,magnet

end;

nU = 40
U = range(0.0, length=nU,stop=4.0)
U = convert(Array{Float64},U)

nT = 50
T = range(0.0, length=nT,stop=1200)
T = convert(Array{Float64},T)


glocr = SharedArray{ComplexF64}(nω,2,nU,nT)
Σ2 = SharedArray{ComplexF64}(nω,2,nU,nT)

@inbounds @sync @distributed for iU in 1:nU
    for iT in 1:nT
        glocr[:,:,iU,iT],Σ1,Σ2[:,:,iU,iT],magnet =  ipt_selfcons(ω,gloc,D0ω,t,U[iU],T[iT],itermax,nω,zeroplus,mix,tol)
    end
end

phase = -imag(glocr[Int64(4096/2),:,:,:])
phase = reshape(sum(phase,dims=1),40,50)
phase = transpose(phase[:,end:-1:1]);
phase = convert(Array{Float64,2},phase);

using JLD2
@save "ipt_ph_diagram_param.jld2" phase U T

plt.figure(figsize=(10,6))

plt.subplot(1,2,1)
plt.contourf(U,T,phase,cmap="viridis")
plt.xlim(0.,4.0)
plt.ylim(0,1200)
plt.ylabel("T (K)")
plt.xlabel("U (eV)")

plt.subplot(1,2,2)
y = [1200, 1100, 1000, 900, 800, 750, 700, 600, 500, 400, 300, 200, 100, 0]
x1 = [0.0, 0.5, 0.8, 1.1, 1.3, 1.38, 1.35, 1.39, 1.43, 1.45, 1.48, 1.51, 1.52, 1.53]
x2 = [0.0, 0.5, 0.8, 1.1, 1.3, 1.38, 1.65, 1.85, 2.05, 2.2, 2.3, 2.4, 2.5, 2.6]
plt.plot(4.0 .- x1,y,"-o",color="blue")
plt.plot(4.0 .- x2,y,"-o",color="blue")
plt.plot(4.0 .- 1.38,750,"-o",color="red")
plt.xlim(0.,4.0)
plt.ylim(0,1200)
plt.xlabel("U (eV)")

plt.text(0.6,500,"Logam",fontsize=14)
plt.text(1.7,150,"Daerah")
plt.text(1.55,100,"Koeksistensi",fontsize=9)
plt.text(2.8,300,"Isolator",fontsize=14)
plt.text(1.6, 750, "Tc ≈ 750 K")
plt.text(1.6, 790, "Uc ≈ 1.38 eV")

plt.show()
plt.savefig("ipt_ph_diagram_param.pdf",format="pdf")
