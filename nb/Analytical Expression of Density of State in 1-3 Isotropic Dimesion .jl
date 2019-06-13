
using PyPlot
using Elliptic
using PyCall
np = pyimport("numpy");

#input parameter
nω = 2^9
ωmin = -3.0
ωmax = 3.0
t = 0.5;

#mesh
ω = range(ωmin,length=nω,stop=ωmax)
ω = convert(Array{Float64,1},ω);

#dos in 1D
function dos1D(t::Float64,ω::Float64)
    sqr = 4*t^2 - ω^2
    if sqr > 0.
        return 1 / sqrt(sqr) / π
    else
        return 0.0
    end
end;

#dos in 2D
function dos2D(t::Float64,ω::Float64)
    m_a = 1. - (ω / (2. * t))^2 / 4.
    if m_a < 0. || m_a > 1.f0
        return 0.0
    else
        return Elliptic.K.(m_a) / (2. * t * π^2) 
    end
end;

#dos in 3D
function dos3D(t::Float64,ω::Float64)
    z = range(-0.999,length=250,stop=0.999)
    z = convert(Array{Float64,1},z);
    ma = 1. .- ((ω / (2. * t) .+ z) ./ 2.).^2.
    km = zeros(Float64,length(ma))
    for (imax,nma) in enumerate(ma)
        if nma > 0. && nma < 1.
            km[imax] = Elliptic.K.(nma)
        end
    end
    intx = km ./ sqrt.(1. .- z.^2)
    return np.trapz(intx,z) / (π^3. * 2. * t)
end;

#dos in bethe
function dosbethe(t::Float64,ω::Float64)
    if ω^2 < 4*t^2
        return (sqrt(4*t^2 - ω^2) / (2*π*t^2))
    else 
        return 0.0
    end
end;

@time A1D = dos1D.(t,ω);

@time A2D = dos2D.(t,ω);

@time A3D = dos3D.(t,ω);

@time Abethe = dosbethe.(t,ω);

plt.figure(1,(7,4))
plt.plot(ω,A1D,label="1D")
plt.plot(ω,A2D .+ 0.4 * 1,label="2D")
plt.plot(ω,A3D .+ 0.4 * 2,label="3D")
plt.plot(ω,Abethe .+ 0.4 * 3,label="bethe")
plt.plot(ω,ones(length(ω)) .* 0.4, color = "black", linewidth = 0.5)
plt.plot(ω,ones(length(ω)) .* 0.4 * 2, color = "black", linewidth = 0.5)
plt.plot(ω,ones(length(ω)) .* 0.4 * 3, color = "black", linewidth = 0.5)
plt.ylim(0,2.0)
plt.xlim(ωmin,ωmax)
plt.ylabel("ρ(ω)")
plt.xlabel("ω (eV)")
plt.title("Density Of State for 1-3 isotropic dimension")
plt.legend()
plt.show()

println("Check Normalization (must be ≈ 1)")
println("")
@show np.trapz(A1D,ω)
@show np.trapz(A2D,ω)
@show np.trapz(A3D,ω)
@show np.trapz(Abethe,ω);

plt.plot(ω,A3D,label="Kubik")
plt.plot(ω,Abethe,label="Bethe")
plt.ylim(0,0.7)
plt.xlim(ωmin,ωmax)
plt.ylabel("D(ω)")
plt.xlabel("ω (eV)")
plt.legend()
plt.show()
plt.savefig("bethe-kubik.pdf",format="pdf")


