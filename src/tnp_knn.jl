
struct KNNTNP <: TNPType
    k::Int
end

function _predict(
    mode::KNNTNP,
    model::TNPModel,
    context_x::AbstractMatrix{<:Real}, 
    context_y::AbstractMatrix{<:Real}, 
    target_x::AbstractVector{<:Real},
)
    # Find k nearest neighbors
    k = mode.k
    distances = sqrt.(sum.(eachcol((context_x .- target_x) .^ 2)))
    sorted_indices = sortperm(distances)
    knn_indices = sorted_indices[1:min(k, length(sorted_indices))]
    
    # Filter context to only include k nearest neighbors
    knn_context_x = context_x[:, knn_indices]
    knn_context_y = context_y[:, knn_indices]
    
    # Call StandardTNP predict with the filtered context and single target point
    mean, std = _predict(StandardTNP(), model, knn_context_x, knn_context_y, hcat(target_x))
    return mean[:,1], std[:,1]
end

function _predict(
    mode::KNNTNP,
    model::TNPModel,
    context_x::AbstractArray{<:Real, 3}, 
    context_y::AbstractArray{<:Real, 3}, 
    target_x::AbstractArray{<:Real, 3},
)
    # context_x, context_y, target_x have shape (dim, N, batch)
    batch_size = size(target_x, 3)
    num_targets = size(target_x, 2)
    y_dim = size(context_y, 1)
    
    # Preallocate result arrays
    means_3d = zeros(y_dim, num_targets, batch_size)
    stds_3d = zeros(y_dim, num_targets, batch_size)
    
    for b in 1:batch_size
        for i in 1:num_targets
            # Extract 2D slices for this batch and target point
            ctx_x = context_x[:, :, b]
            ctx_y = context_y[:, :, b]
            tgt_x = target_x[:, i, b]
            
            # Call the 2D/1D version
            mean, std = _predict(mode, model, ctx_x, ctx_y, tgt_x)
            
            means_3d[:, i, b] = mean
            stds_3d[:, i, b] = std
        end
    end
    
    return means_3d, stds_3d
end
