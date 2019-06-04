module TensorIntegration
    export make_tensor, tensor_integrate
    
    function simpson(a,b,N)
        if mod(N,2) == 0
            N += 1
        end
        if N > 1
            xs = collect(range(a, b, length=N))
            ws = ones(N)
            ws[2:2:end-1].=4
            ws[3:2:end-2].=2
            ws = ws*(xs[2]-xs[1])/3.
        else
            xs = [(a+b)/2.]
            ws = [1.] 
        end
        return xs, ws
    end

    function tensor_simpson(a,b,N; lowprecision = false)
        if lowprecision
            numbertype = Float32
        else
            numbertype = Float64
        end
        d = length(N)
        for i = 1:d
            if mod(N[i],2) == 0
                N[i] += 1
            end
        end

        xs = Array{Vector{numbertype}}(d)
        for i in 1:d
            if N[i] > 1
                xs[i] = collect(range(a[i], b[i], length = N[i]))
            else
                xs[i] = [(a[i]+b[i])/2.]
            end
        end
        ws = Array{Vector{numbertype}}(d)
        for i in 1:d
            if N[i] > 1
                ws[i] = ones(N[i])
                ws[i][2:2:end-1]=4
                ws[i][3:2:end-2]=2
                ws[i] = ws[i]*(xs[i][2]-xs[i][1])/3.
            else
                ws[i] = [1.]
            end
        end
        xmat, w = tensor_integrate(xs, ws, N)
        return xmat, w
    end

    function tensor_integrate(xs, ws, N)
        totN = prod(N)
        d = length(N)
        xmat = Array{eltype(xs[1])}(totN,d)
        wmat = Array{eltype(ws[1])}(totN,d)
        for i in 1:d
            repout = Int(totN/prod(N[1:i]))
            repin = Int(totN/N[i]/repout)
            xmat[:, i] = repeat(xs[i], inner=repin, outer=repout)
            wmat[:, i] = repeat(ws[i], inner=repin, outer=repout)
        end
        w = prod(wmat, 2)
        return xmat, w
    end

    function make_tensor(xs, N)
        totN = prod(N)
        d = length(N)
        xmat = Array{eltype(xs[1])}(totN,d)
        for i in 1:d
            repout = Int(totN/prod(N[1:i]))
            repin = Int(totN/N[i]/repout)
            xmat[:, i] = repeat(xs[i], inner=repin, outer=repout)
        end
        return xmat
    end
end

