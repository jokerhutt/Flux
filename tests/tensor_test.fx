#import "standard.fx", "tensors.fx";

using standard::io::console;
using standard::tensors;

def main() -> int
{
    // --- 1. Construction ---
    I32Array shape1 = i32array_new(2);
    i32array_push(@shape1, 3);
    i32array_push(@shape1, 3);

    Tensor<float> a = tensor_zeros<float>(shape1);
    Tensor<float> b = tensor_ones<float>(shape1);

    // --- 2. Manual fill: a = [[1,2,3],[4,5,6],[7,8,9]] ---
    for (i64 i; i < a.total_size; i = i + 1)
    {
        a.tdata[i] = (float)(i + 1);
    };

    // --- 3. Element-wise add ---
    Tensor<float> c = tensor_add(@a, @b);

    println("tensor_add result (expect 2,3,4,5,6,7,8,9,10):");
    for (i64 i; i < c.total_size; i = i + 1)
    {
        print(f"{c.tdata[i]} \0");
    };
    println("");

    // --- 4. Reductions ---
    float s = tensor_sum_all(@a);
    println(f"tensor_sum_all of a (expect 45): {s}");

    float mn = tensor_min_all(@a);
    float mx = tensor_max_all(@a);
    println(f"tensor_min_all (expect 1): {mn}");
    println(f"tensor_max_all (expect 9): {mx}");

    float mean = tensor_mean(@a);
    println(f"tensor_mean (expect 5): {mean}");

    // --- 5. tensor_at / tensor_set_at ---
    i32[2] coords;
    coords[0] = 1;
    coords[1] = 2;
    float val = tensor_at(@a, coords);
    println(f"tensor_at [1][2] of a (expect 6): {val}");

    tensor_set_at(@a, coords, 99.0f);
    val = tensor_at(@a, coords);
    println(f"tensor_at [1][2] after set_at 99 (expect 99): {val}");
    tensor_set_at(@a, coords, 6.0f);

    // --- 6. tensor_copy ---
    Tensor<float> a_copy = tensor_copy(@a);
    a_copy.tdata[0] = 999.0f;
    println(f"original a[0] after copy mutation (expect 1): {a.tdata[0]}");
    println(f"copy a[0] after mutation (expect 999): {a_copy.tdata[0]}");

    // --- 7. tensor_fill ---
    tensor_fill(@b, 7.0f);
    println(f"tensor_fill 7: b[4] (expect 7): {b.tdata[4]}");

    // --- 8. tensor_arange ---
    Tensor<float> r = tensor_arange<float>(0.0f, 5.0f, 1.0f);
    println("tensor_arange 0..5 step 1 (expect 0 1 2 3 4):");
    for (i64 i; i < r.total_size; i = i + 1)
    {
        print(f"{r.tdata[i]} \0");
    };
    println("");

    // --- 9. tensor_sum_axis ---
    I32Array shape2 = i32array_new(2);
    i32array_push(@shape2, 2);
    i32array_push(@shape2, 3);
    Tensor<float> m = tensor_zeros<float>(shape2);
    // [[1,2,3],[4,5,6]]
    for (i64 i; i < m.total_size; i = i + 1)
    {
        m.tdata[i] = (float)(i + 1);
    };
    Tensor<float> row_sums = tensor_sum_axis(@m, 1);
    println("tensor_sum_axis axis=1 of [[1,2,3],[4,5,6]] (expect 6 15):");
    for (i64 i; i < row_sums.total_size; i = i + 1)
    {
        print(f"{row_sums.tdata[i]} \0");
    };
    println("");

    // --- 10. tensor_matmul ---
    I32Array shape3 = i32array_new(2);
    i32array_push(@shape3, 2);
    i32array_push(@shape3, 2);
    Tensor<float> p = tensor_zeros<float>(shape3);
    Tensor<float> q = tensor_zeros<float>(shape3);
    // p = [[1,2],[3,4]], q = [[5,6],[7,8]]
    p.tdata[0] = 1.0f; p.tdata[1] = 2.0f;
    p.tdata[2] = 3.0f; p.tdata[3] = 4.0f;
    q.tdata[0] = 5.0f; q.tdata[1] = 6.0f;
    q.tdata[2] = 7.0f; q.tdata[3] = 8.0f;
    Tensor<float> pq = tensor_matmul(@p, @q);
    println("tensor_matmul [[1,2],[3,4]] x [[5,6],[7,8]] (expect 19 22 43 50):");
    for (i64 i; i < pq.total_size; i = i + 1)
    {
        print(f"{pq.tdata[i]} \0");
    };
    println("");

    // --- 11. tensor_tensordot ---
    // Contract axis 1 of p (cols) with axis 0 of q (rows): same as matmul for 2D
    Tensor<float> td = tensor_tensordot(@p, @q, 1, 0);
    println("tensor_tensordot p axis1 x q axis0 (expect 19 22 43 50):");
    for (i64 i; i < td.total_size; i = i + 1)
    {
        print(f"{td.tdata[i]} \0");
    };
    println("");

    // --- 12. tensor_reshape ---
    I32Array flat_shape = i32array_new(1);
    i32array_push(@flat_shape, 4);
    Tensor<float> pq_flat = tensor_reshape(@pq, flat_shape);
    println("tensor_reshape 2x2 -> 4 (expect 19 22 43 50):");
    for (i64 i; i < pq_flat.total_size; i = i + 1)
    {
        print(f"{pq_flat.tdata[i]} \0");
    };
    println("");

    // --- 13. tensor_transpose ---
    I32Array axes = i32array_new(2);
    i32array_push(@axes, 1);
    i32array_push(@axes, 0);
    Tensor<float> pt = tensor_transpose(@p, axes);
    println("tensor_transpose [[1,2],[3,4]] (expect 1 3 2 4):");
    for (i64 i; i < pt.total_size; i = i + 1)
    {
        print(f"{pt.tdata[i]} \0");
    };
    println("");

    // --- 14. integer tensors ---
    I32Array ishape = i32array_new(1);
    i32array_push(@ishape, 4);
    Tensor<i32> iv = tensor_arange<i32>(10, 14, 1);
    println("integer tensor_arange 10..14 (expect 10 11 12 13):");
    for (i64 i; i < iv.total_size; i = i + 1)
    {
        print(f"{iv.tdata[i]} \0");
    };
    println("");

    // --- Cleanup ---
    tensor_free(@a);
    tensor_free(@b);
    tensor_free(@c);
    tensor_free(@a_copy);
    tensor_free(@r);
    tensor_free(@m);
    tensor_free(@row_sums);
    tensor_free(@p);
    tensor_free(@q);
    tensor_free(@pq);
    tensor_free(@td);
    tensor_free(@pq_flat);
    tensor_free(@pt);
    tensor_free(@iv);
    i32array_free(@shape1);
    i32array_free(@shape2);
    i32array_free(@shape3);
    i32array_free(@axes);
    i32array_free(@ishape);

    return 0;
};
