module num

using DSP

export trapz, conv_same, convergent, mixing

function trapz(x,y)
    dx = x[2] - x[1]
    intgrl = 0.0
    @fastmath @inbounds for i in 1:length(y)-1
        intgrl += (y[i+1] + y[i])
    end
    return intgrl * 0.5 * dx
end


function conv_same(A,B)
    C = conv(A,B)
    lower_bound = floor(Int,length(C)/4)
    upper_bound = floor(Int,3*length(C)/4-1)
    lengthc = upper_bound-lower_bound + 1
    if lengthc != length(A)
        upper_bound = floor(Int,3*length(C)/4)
    end
    return C[lower_bound:upper_bound]
end

function convergent(value_old,value_new,ω,nω,tol)
    error = zeros(Float64,nω)
    if length(value_new) != nω
        for i = 1:nω
            error[i] = abs( sum( imag.(value_new[i,:]) .- imag.(value_old[i,:]) ) )
        end
    else
        for i = 1:nω
            error[i] = abs( imag.(value_new[i]) .- imag.(value_old[i]) )
        end
    end
    err = trapz(ω,error)
    convg = err <= tol
    return convg,err
end

function mixing(value_old,value_new,mixings)
    return (1 - mixings) .* value_old .+ mixings .* value_new
end

end
