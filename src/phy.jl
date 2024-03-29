module physics

include("num.jl")
using .num
using Elliptic

export dos3D, dosbethe, baredos, fermi, zeta

function dos3D(t::Float64,ω::Float64)
    z = range(-0.995,length=250,stop=0.995)
    z = convert(Array{Float64,1},z);
    ma = 1. .- ((ω / (2. * t) .+ z) ./ 2.).^2.
    km = zeros(Float64,length(ma))
    for (imax,nma) in enumerate(ma)
        if nma > 0. && nma < 1.
            km[imax] = Elliptic.K.(nma)
        end
    end
    intx = km ./ sqrt.(1. .- z.^2)
    return num.trapz(z,intx) / (π^3. * 2. * t)
end

function dosbethe(t::Float64,ω::Float64)
    if ω^2 < 4*t^2
        return (sqrt(4*t^2 - ω^2) / (2*π*t^2))
    else 
        return 0.0
    end
end

function baredos(typedos::String,t::Float64,ω::Float64)
    if typedos == "cubic"
        return dos3D(t,ω)
    elseif typedos == "bethe"
        return dosbethe(t,ω)
    else
        println(error("current typedos is either cubic or bethe"))
    end
end

function fermi(ω::Float64,T::Float64)
    kB = 8.61733034e-5;
    if T==0
        if ω <= 0
            return 1.0
        else
            return 0.0
        end
    else
        β = 1. / kB / T
        return 1. ./ ( exp(ω*β) + 1.)
    end
end

zeta(ω,zeroplus) = ω + im * zeroplus

end
