## Foundational tensor operations in base R (ranks 2-5).
## Tensor2 maps to a matrix, higher ranks to N-D arrays. Rank-1 (Vector)
## lives in vector-r/R/vector.R.

morloc_zeros2 <- function(d1, d2) array(0, dim = c(d1, d2))
morloc_zeros3 <- function(d1, d2, d3) array(0, dim = c(d1, d2, d3))
morloc_zeros4 <- function(d1, d2, d3, d4) array(0, dim = c(d1, d2, d3, d4))
morloc_zeros5 <- function(d1, d2, d3, d4, d5) array(0, dim = c(d1, d2, d3, d4, d5))

morloc_ones2 <- function(d1, d2) array(1, dim = c(d1, d2))
morloc_ones3 <- function(d1, d2, d3) array(1, dim = c(d1, d2, d3))
morloc_ones4 <- function(d1, d2, d3, d4) array(1, dim = c(d1, d2, d3, d4))
morloc_ones5 <- function(d1, d2, d3, d4, d5) array(1, dim = c(d1, d2, d3, d4, d5))

morloc_fill2 <- function(v, d1, d2) array(v, dim = c(d1, d2))
morloc_fill3 <- function(v, d1, d2, d3) array(v, dim = c(d1, d2, d3))
morloc_fill4 <- function(v, d1, d2, d3, d4) array(v, dim = c(d1, d2, d3, d4))
morloc_fill5 <- function(v, d1, d2, d3, d4, d5) array(v, dim = c(d1, d2, d3, d4, d5))

morloc_identity <- function(n) diag(n)

morloc_matmul <- function(a, b) a %*% b


## Packable: tuple-of(dims, flat array) <-> shaped array.
## morloc represents tuples as lists, so packed = list(dims_list, data_array).
## Vector itself has no pack/unpack: morloc Vector maps directly to R's
## native (1-D) array, so the wire form is the runtime form.
##
## Storage-order convention: the wire format is row-major (same as
## C++ mlc::Tensor and numpy default), but R stores multi-dim arrays
## column-major. Pack and unpack must therefore reverse the axis order
## with aperm so that semantic indices `r[i,j,...] == t(i-1,j-1,...)`
## match across language boundaries. Vector (rank 1) is unaffected.

## Build an R column-major array from a row-major flat buffer.
.morloc_from_rowmajor <- function(data, dims) {
    rank <- length(dims)
    aperm(array(data, dim = rev(dims)), rev(seq_len(rank)))
}

## Linearize an R column-major array as a row-major flat buffer.
.morloc_to_rowmajor <- function(t) {
    rank <- length(dim(t))
    as.vector(aperm(t, rev(seq_len(rank))))
}

morloc_packMatrix <- function(packed) {
    dims <- packed[[1]]; data <- packed[[2]]
    .morloc_from_rowmajor(data, c(dims[[1]], dims[[2]]))
}

morloc_unpackMatrix <- function(t) {
    list(list(dim(t)[1], dim(t)[2]), .morloc_to_rowmajor(t))
}

morloc_packTensor3 <- function(packed) {
    dims <- packed[[1]]; data <- packed[[2]]
    .morloc_from_rowmajor(data, c(dims[[1]], dims[[2]], dims[[3]]))
}

morloc_unpackTensor3 <- function(t) {
    list(list(dim(t)[1], dim(t)[2], dim(t)[3]), .morloc_to_rowmajor(t))
}

morloc_packTensor4 <- function(packed) {
    dims <- packed[[1]]; data <- packed[[2]]
    .morloc_from_rowmajor(data, c(dims[[1]], dims[[2]], dims[[3]], dims[[4]]))
}

morloc_unpackTensor4 <- function(t) {
    list(list(dim(t)[1], dim(t)[2], dim(t)[3], dim(t)[4]), .morloc_to_rowmajor(t))
}

morloc_packTensor5 <- function(packed) {
    dims <- packed[[1]]; data <- packed[[2]]
    .morloc_from_rowmajor(data, c(dims[[1]], dims[[2]], dims[[3]], dims[[4]], dims[[5]]))
}

morloc_unpackTensor5 <- function(t) {
    list(list(dim(t)[1], dim(t)[2], dim(t)[3], dim(t)[4], dim(t)[5]), .morloc_to_rowmajor(t))
}
