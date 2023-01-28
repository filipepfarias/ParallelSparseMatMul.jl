### Initialization functions
"""
   shsprand(m,n,p)

Initialize randomly a  `m` by `n` `SharedSparseMatrixCSC` with density factor `p`

# Examples
```julia-repl
julia> Matrix(SharedSparseMatMul.shsprand(4,4,0.7))
4×4 Array{Float64,2}:
0.941088  0.0        0.0       0.0
0.0       0.0132652  0.580678  0.0192514
0.0       0.0        0.566281  0.0
0.0       0.0        0.0       0.6452
```

"""

function shsprand(m,n,p; kwargs...)
    colptr = SharedArray{Int64}(n+1; kwargs...)
    colptr[1] = 1
    for i=2:n+1
        inc = round(p*m+sqrt(m*p*(1-p))*randn())
        colptr[i] = colptr[i-1]+max(1,inc)
    end
    nnz = colptr[end]-1
    nzval = SharedArrays.shmem_rand(nnz; kwargs...)
    # multiplication will go faster if you sort these within each column...
    rowval = shmem_randsample(nnz,1,m;sorted_within=colptr, kwargs...)
    return SharedSparseMatrixCSC(m,n,colptr,rowval,nzval)
end

function shsprandn(m,n,p; kwargs...)
    colptr = SharedArray(Int64,n+1; kwargs...)
    colptr[1] = 1
    for i=2:n+1
        inc = round(p*m+sqrt(m*p*(1-p))*randn())
        colptr[i] = colptr[i-1]+max(1,inc)
    end
    nnz = colptr[end]-1
    nzval = SharedArrays.shmem_randn(nnz; kwargs...)
    rowval = SharedArrays.shmem_randsample(nnz,1,m;sorted_within=colptr, kwargs...)
    return SharedSparseMatrixCSC(m,n,colptr,rowval,nzval)
end

function shmem_randsample(n,minval,maxval;sorted_within=[], kwargs...)
    out = SharedArrays.shmem_rand(minval:maxval,n; kwargs...)
    # XXX do this in parallel ONLY ON PARTICIPATING WORKERS
    @distributed for i=2:length(sorted_within)
        out[sorted_within[i-1]:sorted_within[i]-1] = sort(out[sorted_within[i-1]:sorted_within[i]-1])
    end
    return out
end

"""
    shspeye(T::Type, m::Integer, n:Integer)

Build a `SharedSparseMatrix` containing a *identity matrix* of size `m` by `n`
```julia-repl
julia> shspeye(Float64, 4,3)
4×3 SparseArrays.SparseMatrixCSC{Float64,Int64} with 3 stored entries:
  [1, 1]  =  1.0
  [2, 2]  =  1.0
  [3, 3]  =  1.0
```


"""
function shspeye(T::Type, m::Integer, n::Integer)
    x = min(m,n)
    rowval = share(collect(1:x))
    colptr = share([rowval; fill(x+1, n+1-x)]) # find why int is necessary in the old version
    nzval  = SharedArrays.shmem_fill(one(T),x)
    return SharedSparseMatrixCSC(m, n, colptr, rowval, nzval)
end

shspeye(n::Integer) = shspeye(Float64,n,n)
