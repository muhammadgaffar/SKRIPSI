module util
    using DSP   
    using Dierckx     
    using PyCall

    np = pyimport("numpy")

    export simpson,conv_same,convergent,mixing,smoothing

    function simpson(params,n)
        xmin = params["ωrange"][1]
        xmax = params["ωrange"][2]

        h = (xmax-xmin)/n
        weight = zeros(Float64,n)

        weight[1] = h/3
        for i in 2:2:n-1
            weight[i] = 4h/3
        end
        for i in 3:2:n-2
            weight[i] = 2h/3
        end
        weight[n] = h/3
        return weight
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

    function convergent(value_old,value_new,params)
        nω = params["nω"]
        tol = params["tolerance"]
        error = zeros(Float64,nω)
        ww = util.simpson(params,nω)
        if length(value_new) != nω
            for i = 1:nω
                error[i] = abs( sum( imag.(value_new[i,:]) .- imag.(value_old[i,:]) ) )
            end
        else
            for i = 1:nω
                error[i] = abs( imag.(value_new[i]) .- imag.(value_old[i]) )
            end
        end
        err = 0.0
        for i =1:nω
            err += error[i]*ww[i]
        end
        convg = err <= tol
        return convg,err
    end

    function mixing(value_old,value_new,mixings)
        return (1 - mixings) .* value_old .+ mixings .* value_new
    end

    function smoothing(ω,gloc,vd)
        a,b = np.shape(gloc)
        smooth = zeros(ComplexF64,floor(Int,a/vd),b)
        for i = 1:b
            smooth[:,i] = real(gloc[1:vd:a,i]) .+ im.*imag(gloc[1:vd:a,i])
        end
        wx = ω[1:vd:a]
        for i = 1:b
            splreal = Spline1D(wx,real(smooth[:,i]))
            splimag = Spline1D(wx,imag(smooth[:,i]))
            for j = 1:a
                gloc[j,i] = splreal(ω[j]) + im * splimag(ω[j])
            end
        end
        return gloc
    end

    function real2matsubara(wn,g_iwn,ω,gloc,params)
        #minimal cost integration
        ww = util.simpson(params,params["nω"])
        
        _,dim = np.shape(gloc)
        for d = 1:dim
            for i = 1:length(wn)
                dum = 0
                for j = 1:params["nω"]
                    dum += ww[j] .* -1/π .* imag(gloc[j,d]) ./ (im*wn[i]-ω[j])
                end
                g_iwn[i,d] = dum
            end
        end
        return g_iwn
    end

end