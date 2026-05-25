// tensors.fx – Multi‑dimensional array (Tensor) library.
// Conforms to Flux language specification and style guide.
// No variable declarations inside loops – all at function top.
// Loop counters declared inside for parentheses.
// Zero initialization is automatic.

#ifndef FLUX_STANDARD_TENSORS
#def FLUX_STANDARD_TENSORS 1;

#ifndef FLUX_STANDARD_TYPES
#import "types.fx";
#endif;

#ifndef FLUX_STANDARD_MATH
#import "math.fx";
#endif;

using standard::math;

namespace standard
{
    namespace tensors
    {
        ///  ---------------------------------------------------------------
             DYNAMIC INTEGER ARRAY (SHAPE AND STRIDES)
        ///

        struct I32Array
        {
            i32* adata;
            i32  len, cap;
        };

        def i32array_new(i32 capacity) -> I32Array
        {
            I32Array a;
            size_t bytes = (size_t)(capacity * (i32)sizeof(i32));
            a.len = 0;
            a.cap = capacity;
            a.adata = (@)fmalloc(bytes);
            return a;
        };

        def i32array_free(I32Array* a) -> void
        {
            if (a.adata != (i32*)0)
            {
                ffree((u64)a.adata);
                a.adata = (i32*)0;
            };
            a.len = 0;
            a.cap = 0;
        };

        def i32array_push(I32Array* a, i32 value) -> void
        {
            if (a.len >= a.cap)
            {
                i32 new_cap = (a.cap == 0) ? 4 : a.cap * 2;
                size_t new_bytes = (size_t)(new_cap * (i32)sizeof(i32));
                i32* new_data = (@)fmalloc(new_bytes);
                for (i32 i; i < a.len; i = i + 1)
                {
                    new_data[i] = a.adata[i];
                };
                ffree((u64)a.adata);
                a.adata = new_data;
                a.cap = new_cap;
            };
            a.adata[a.len] = value;
            a.len = a.len + 1;
        };

        ///  ---------------------------------------------------------------
             TENSOR<T> STRUCT
        ///

        struct Tensor<T>
        {
            T*       tdata;
            i32      ndim;
            I32Array shape, strides;
            i64      total_size;
        };

        // -----------------------------------------------------------------
        //  Internal helpers
        // -----------------------------------------------------------------

        def _tensor_compute_size(I32Array* shape) -> i64
        {
            i64 sz = 1;
            for (i32 i; i < shape.len; i = i + 1)
            {
                sz = sz * (i64)shape.adata[i];
            };
            return sz;
        };

        def _tensor_default_strides(I32Array* shape) -> I32Array
        {
            I32Array strides = i32array_new(shape.len);
            i32 stride = 1;
            for (i32 i = shape.len - 1; i >= 0; i = i - 1)
            {
                i32array_push(@strides, stride);
                stride = stride * shape.adata[i];
            };
            i32 tmp;
            for (i32 i; i < strides.len / 2; i = i + 1)
            {
                tmp = strides.adata[i];
                strides.adata[i] = strides.adata[strides.len - 1 - i];
                strides.adata[strides.len - 1 - i] = tmp;
            };
            return strides;
        };

        def _tensor_index<T>(Tensor<T>* t, i32* coords) -> i64
        {
            i64 idx;
            for (i32 i; i < t.ndim; i = i + 1)
            {
                idx = idx + (i64)coords[i] * (i64)t.strides.adata[i];
            };
            return idx;
        };

        ///  ---------------------------------------------------------------
             MEMORY MANAGEMENT
        ///

        def tensor_free<T>(Tensor<T>* t) -> void
        {
            if (t.tdata != (T*)0)
            {
                ffree((u64)t.tdata);
                t.tdata = (T*)0;
            };
            i32array_free(@t.shape);
            i32array_free(@t.strides);
            t.ndim = 0;
            t.total_size = 0;
        };

        ///  ---------------------------------------------------------------
             CONSTRUCTION
        ///

        def tensor_zeros<T>(I32Array shape) -> Tensor<T>
        {
            Tensor<T> t;
            t.ndim = shape.len;
            t.shape = shape;
            t.strides = _tensor_default_strides(@shape);
            t.total_size = _tensor_compute_size(@shape);
            size_t bytes = (size_t)(t.total_size * (i64)sizeof(T));
            t.tdata = (T*)fmalloc(@bytes);
            for (i64 i; i < t.total_size; i = i + 1)
            {
                t.tdata[i] = (T)0;
            };
            return t;
        };

        def tensor_ones<T>(I32Array shape) -> Tensor<T>
        {
            Tensor<T> t = tensor_zeros<T>(shape);
            for (i64 i; i < t.total_size; i = i + 1)
            {
                t.tdata[i] = (T)1;
            };
            return t;
        };

        def tensor_full<T>(I32Array shape, T value) -> Tensor<T>
        {
            Tensor<T> t = tensor_zeros<T>(shape);
            for (i64 i; i < t.total_size; i = i + 1)
            {
                t.tdata[i] = value;
            };
            return t;
        };

        def tensor_arange<T>(T start, T end, T step) -> Tensor<T>
        {
            i64 n = (i64)((end - start) / step);
            if (n < 0) {n = 0;};
            I32Array shape = i32array_new(1);
            i32array_push(@shape, (i32)n);
            Tensor<T> t = tensor_zeros<T>(shape);
            for (i64 i; i < n; i = i + 1)
            {
                t.tdata[i] = start + (T)i * step;
            };
            i32array_free(@shape);
            return t;
        };

        def tensor_from_array<T, N>(T[N] array) -> Tensor<T>
        {
            I32Array shape = i32array_new(1);
            i32array_push(@shape, N);
            Tensor<T> t = tensor_zeros<T>(shape);
            for (i32 i; i < N; i = i + 1)
            {
                t.tdata[i] = array[i];
            };
            i32array_free(@shape);
            return t;
        };

        ///  ---------------------------------------------------------------
             INDEXING
        ///

        def tensor_at<T>(Tensor<T>* t, i32* coords) -> T
        {
            i64 idx = _tensor_index(t, coords);
            return t.tdata[idx];
        };

        def tensor_set_at<T>(Tensor<T>* t, i32* coords, T value) -> void
        {
            i64 idx = _tensor_index(t, coords);
            t.tdata[idx] = value;
        };

        ///  ---------------------------------------------------------------
             COPY & FILL
        ///

        def tensor_copy<T>(Tensor<T>* src) -> Tensor<T>
        {
            Tensor<T> dst = tensor_zeros<T>(src.shape);
            for (i64 i; i < src.total_size; i = i + 1)
            {
                dst.tdata[i] = src.tdata[i];
            };
            return dst;
        };

        def tensor_fill<T>(Tensor<T>* t, T value) -> void
        {
            for (i64 i; i < t.total_size; i = i + 1)
            {
                t.tdata[i] = value;
            };
        };

        ///  ---------------------------------------------------------------
             RESHAPE
        ///

        def tensor_reshape<T>(Tensor<T>* t, I32Array new_shape) -> Tensor<T>
        {
            i64 new_size = _tensor_compute_size(@new_shape);
            if (new_size != t.total_size)
            {
                Tensor<T> empty;
                empty.tdata = (T*)0;
                empty.ndim = 0;
                return empty;
            };
            Tensor<T> out;
            out.ndim = new_shape.len;
            out.shape = new_shape;
            out.strides = _tensor_default_strides(@new_shape);
            out.total_size = t.total_size;
            size_t bytes = (size_t)(out.total_size * (i64)sizeof(T));
            out.tdata = (T*)fmalloc(@bytes);
            for (i64 i; i < out.total_size; i = i + 1)
            {
                out.tdata[i] = t.tdata[i];
            };
            return out;
        };

        ///  ---------------------------------------------------------------
             TRANSPOSE
        ///

        def tensor_transpose<T>(Tensor<T>* t, I32Array axes) -> Tensor<T>
        {
            if (axes.len != t.ndim) { return tensor_zeros<T>(t.shape); };
            I32Array new_shape = i32array_new(t.ndim);
            for (i32 i; i < axes.len; i = i + 1)
            {
                i32array_push(@new_shape, t.shape.adata[axes.adata[i]]);
            };
            Tensor<T> out = tensor_zeros<T>(new_shape);
            I32Array new_strides = i32array_new(t.ndim);
            i32 stride = 1;
            for (i32 i = t.ndim - 1; i >= 0; i = i - 1)
            {
                i32array_push(@new_strides, stride);
                stride = stride * new_shape.adata[i];
            };
            i32 tmp;
            for (i32 i; i < new_strides.len / 2; i = i + 1)
            {
                tmp = new_strides.adata[i];
                new_strides.adata[i] = new_strides.adata[new_strides.len - 1 - i];
                new_strides.adata[new_strides.len - 1 - i] = tmp;
            };
            i32array_free(@out.strides);
            out.strides = new_strides;
            i32[8] coords_out, coords_src;
            i64 rem;
            for (i64 flat; flat < out.total_size; flat = flat + 1)
            {
                rem = flat;
                for (i32 i; i < out.ndim; i = i + 1)
                {
                    coords_out[i] = (i32)(rem % (i64)out.shape.adata[i]);
                    rem = rem / (i64)out.shape.adata[i];
                };
                for (i32 i; i < axes.len; i = i + 1)
                {
                    coords_src[axes.adata[i]] = coords_out[i];
                };
                out.tdata[flat] = tensor_at(t, coords_src);
            };
            i32array_free(@new_shape);
            return out;
        };

        ///  ---------------------------------------------------------------
             BROADCASTING SHAPE
        ///

        def _broadcast_shape(I32Array* a, I32Array* b) -> I32Array
        {
            i32 nd = (a.len > b.len) ? a.len : b.len;
            I32Array out_shape = i32array_new(nd);
            i32 ai, bi;
            for (i32 i; i < nd; i = i + 1)
            {
                ai = (i < a.len) ? a.adata[a.len - 1 - i] : 1;
                bi = (i < b.len) ? b.adata[b.len - 1 - i] : 1;
                if (ai != bi & ai != 1 & bi != 1)
                {
                    i32array_free(@out_shape);
                    out_shape.adata = (i32*)0;
                    out_shape.len = 0;
                    return out_shape;
                };
                i32array_push(@out_shape, (ai > bi) ? ai : bi);
            };
            i32 tmp;
            for (i32 i; i < out_shape.len / 2; i = i + 1)
            {
                tmp = out_shape.adata[i];
                out_shape.adata[i] = out_shape.adata[out_shape.len - 1 - i];
                out_shape.adata[out_shape.len - 1 - i] = tmp;
            };
            return out_shape;
        };

        ///  ---------------------------------------------------------------
             ELEMENT‑WISE ADD (with broadcasting)
        ///

        def tensor_add<T>(Tensor<T>* a, Tensor<T>* b) -> Tensor<T>
        {
            I32Array out_shape = _broadcast_shape(@a.shape, @b.shape);
            if (out_shape.adata == (i32*)0)
            {
                Tensor<T> empty;
                empty.tdata = (T*)0;
                empty.ndim = 0;
                return empty;
            };
            Tensor<T> out = tensor_zeros<T>(out_shape);
            i32[8] coords;
            i64 idx_a, idx_b;
            i32 dim_a, dim_b, coord;
            i64 rem;
            for (i64 flat; flat < out.total_size; flat = flat + 1)
            {
                rem = flat;
                for (i32 i; i < out.ndim; i = i + 1)
                {
                    coords[i] = (i32)(rem % (i64)out.shape.adata[i]);
                    rem = rem / (i64)out.shape.adata[i];
                };
                idx_a = 0;
                for (i32 i; i < a.ndim; i = i + 1)
                {
                    dim_a = a.shape.adata[i];
                    coord = (i < out.ndim) ? coords[out.ndim - a.ndim + i] : 0;
                    if (dim_a == 1) {coord = 0;};
                    idx_a = idx_a + (i64)coord * (i64)a.strides.adata[i];
                };
                idx_b = 0;
                for (i32 i; i < b.ndim; i = i + 1)
                {
                    dim_b = b.shape.adata[i];
                    coord = (i < out.ndim) ? coords[out.ndim - b.ndim + i] : 0;
                    if (dim_b == 1) {coord = 0;};
                    idx_b = idx_b + (i64)coord * (i64)b.strides.adata[i];
                };
                out.tdata[flat] = a.tdata[idx_a] + b.tdata[idx_b];
            };
            i32array_free(@out_shape);
            return out;
        };

        ///  ---------------------------------------------------------------
             REDUCTIONS
        ///

        def tensor_sum_all<T>(Tensor<T>* t) -> T
        {
            T s;
            for (i64 i; i < t.total_size; i = i + 1)
            {
                s = s + t.tdata[i];
            };
            return s;
        };

        def tensor_sum_axis<T>(Tensor<T>* t, i32 axis) -> Tensor<T>
        {
            if (axis < 0 | axis >= t.ndim) { return tensor_zeros<T>(t.shape); };
            I32Array out_shape = i32array_new(t.ndim - 1);
            for (i32 i; i < axis; i = i + 1)
            {
                i32array_push(@out_shape, t.shape.adata[i]);
            };
            for (i32 i = axis + 1; i < t.ndim; i = i + 1)
            {
                i32array_push(@out_shape, t.shape.adata[i]);
            };
            Tensor<T> out = tensor_zeros<T>(out_shape);
            i32[8] coords;
            i64 idx, flat, rem;
            i32 dim = t.shape.adata[axis],
                stride_axis = t.strides.adata[axis],
                out_idx;
            T sum;
            for (flat = 0; flat < out.total_size; flat = flat + 1)
            {
                rem = flat;
                for (i32 i; i < out.ndim; i = i + 1)
                {
                    out_idx = (i < axis) ? i : i + 1;
                    coords[out_idx] = (i32)(rem % (i64)out.shape.adata[i]);
                    rem = rem / (i64)out.shape.adata[i];
                };
                sum = 0;
                for (i32 k; k < dim; k = k + 1)
                {
                    coords[axis] = k;
                    idx = _tensor_index(t, coords);
                    sum = sum + t.tdata[idx];
                };
                out.tdata[flat] = sum;
            };
            i32array_free(@out_shape);
            return out;
        };

        def tensor_mean<T>(Tensor<T>* t) -> T
        {
            T s = tensor_sum_all(t);
            return s / (T)t.total_size;
        };

        def tensor_min_all<T>(Tensor<T>* t) -> T
        {
            if (t.total_size == 0) { return (T)0; };
            T m = t.tdata[0];
            for (i64 i = 1; i < t.total_size; i = i + 1)
            {
                if (t.tdata[i] < m) {m = t.tdata[i];};
            };
            return m;
        };

        def tensor_max_all<T>(Tensor<T>* t) -> T
        {
            if (t.total_size == 0) { return (T)0; };
            T m = t.tdata[0];
            for (i64 i = 1; i < t.total_size; i = i + 1)
            {
                if (t.tdata[i] > m) {m = t.tdata[i];};
            };
            return m;
        };

        ///  ---------------------------------------------------------------
             MATRIX MULTIPLICATION (2‑D ONLY)
        ///

        def tensor_matmul<T>(Tensor<T>* a, Tensor<T>* b) -> Tensor<T>
        {
            if (a.ndim != 2 | b.ndim != 2) { return tensor_zeros<T>(a.shape); };
            i32 a_rows = a.shape.adata[0];
            i32 a_cols = a.shape.adata[1];
            i32 b_rows = b.shape.adata[0];
            i32 b_cols = b.shape.adata[1];
            if (a_cols != b_rows) {return tensor_zeros<T>(a.shape);};
            I32Array out_shape = i32array_new(2);
            i32array_push(@out_shape, a_rows);
            i32array_push(@out_shape, b_cols);
            Tensor<T> out = tensor_zeros<T>(out_shape);
            i32[2] coords_a, coords_b, coords_out;
            T sum;
            for (i32 i; i < a_rows; i = i + 1)
            {
                for (i32 j; j < b_cols; j = j + 1)
                {
                    sum = 0;
                    for (i32 k; k < a_cols; k = k + 1)
                    {
                        coords_a[0] = i; coords_a[1] = k;
                        coords_b[0] = k; coords_b[1] = j;
                        sum = sum + tensor_at(a, coords_a) * tensor_at(b, coords_b);
                    };
                    coords_out[0] = i; coords_out[1] = j;
                    tensor_set_at(@out, coords_out, sum);
                };
            };
            i32array_free(@out_shape);
            return out;
        };

        ///  ---------------------------------------------------------------
             TENSOR CONTRACTION (tensordot) – one axis pair
        ///

        def tensor_tensordot<T>(Tensor<T>* a, Tensor<T>* b, i32 axis_a, i32 axis_b) -> Tensor<T>
        {
            if (axis_a < 0 | axis_a >= a.ndim | axis_b < 0 | axis_b >= b.ndim) { return tensor_zeros<T>(a.shape); };
            i32 dim = a.shape.adata[axis_a];
            if (dim != b.shape.adata[axis_b]) {return tensor_zeros<T>(a.shape);};
            I32Array out_shape = i32array_new(a.ndim + b.ndim - 2);
            for (i32 i; i < a.ndim; i = i + 1)
            {
                if (i != axis_a) {i32array_push(@out_shape, a.shape.adata[i]);};
            };
            for (i32 i; i < b.ndim; i = i + 1)
            {
                if (i != axis_b) {i32array_push(@out_shape, b.shape.adata[i]);};
            };
            Tensor<T> out = tensor_zeros<T>(out_shape);
            i32[8] coords_a, coords_b, coords_out;
            i32 out_pos;
            i64 idx_a, idx_b, rem;
            T sum;
            for (i64 flat; flat < out.total_size; flat = flat + 1)
            {
                rem = flat;
                for (i32 i; i < out.ndim; i = i + 1)
                {
                    coords_out[i] = (i32)(rem % (i64)out.shape.adata[i]);
                    rem = rem / (i64)out.shape.adata[i];
                };
                // Map output coordinates to input coordinates
                out_pos = 0;
                for (i32 i; i < a.ndim; i = i + 1)
                {
                    if (i == axis_a) {continue;};
                    coords_a[i] = coords_out[out_pos];
                    out_pos = out_pos + 1;
                };
                for (i32 i; i < b.ndim; i = i + 1)
                {
                    if (i == axis_b) {continue;};
                    coords_b[i] = coords_out[out_pos];
                    out_pos = out_pos + 1;
                };
                sum = 0;
                for (i32 k; k < dim; k = k + 1)
                {
                    coords_a[axis_a] = k;
                    coords_b[axis_b] = k;
                    idx_a = _tensor_index(a, coords_a);
                    idx_b = _tensor_index(b, coords_b);
                    sum = sum + a.tdata[idx_a] * b.tdata[idx_b];
                };
                out.tdata[flat] = sum;
            };
            i32array_free(@out_shape);
            return out;
        };
    };
};
#endif;