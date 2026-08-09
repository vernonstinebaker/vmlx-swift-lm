import Cmlx
import MLX

func dynamicSliceUpdate(
    _ source: MLXArray,
    update: MLXArray,
    start: MLXArray,
    axes: [Int32],
    stream: StreamOrDevice = .default
) -> MLXArray {
    var result = mlx_array_new()
    var axes = axes
    let resultCode = mlx_slice_update_dynamic(
        &result,
        source.ctx,
        update.ctx,
        start.ctx,
        &axes,
        axes.count,
        stream.ctx)
    precondition(resultCode == 0, "Dynamic slice update failed")
    return MLXArray(result)
}
