/**
*  TensorView
*  View array data as multidimensional tensors of various shapes efficiently
*  @VERSION 2.0.0
*  https://github.com/foo123/TensorView
*
**/
!function(root, name, factory) {
"use strict";
if ('object' === typeof exports)
    // CommonJS module
    module.exports = factory();
else if ('function' === typeof define && define.amd)
    // AMD. Register as an anonymous module.
    define(function(req) {return factory();});
else
    root[name] = factory();
}('undefined' !== typeof self ? self : this, 'TensorView', function(undef) {
"use strict";

var proto = 'prototype',
    stdMath = Math,
    TypedArray = "undefined" !== typeof Float32Array ? Object.getPrototypeOf(Float32Array) : null;

function TensorView(data, o, _)
{
    if (!(this instanceof TensorView)) return new TensorView(data, o, _);

    var self = this,
        is_transposed = false, is_value = false,
        refs = null, stack = null, stack_axis = -1,
        aux_indices = null, nd_shape = null, same_shape = false,
        ndim = 0, shape = null, stride = null,
        length = 0, total = 0, i;

    function get(index, indices)
    {
        if (refs)
        {
            return refs[0].get(indices);
        }
        else if (stack)
        {
            for (var i=0; i<ndim; ++i)
            {
                aux_indices[i] = indices[i];
            }
            for (var i=0,sl=stack.length,t,tl; i<sl; ++i)
            {
                t = stack[i]; tl = t.shape(stack_axis);
                if (0 <= aux_indices[stack_axis] && aux_indices[stack_axis] < tl) return t.get(aux_indices);
                aux_indices[stack_axis] -= tl;
            }
        }
        else if (nd_shape)
        {
            return walk(data, same_shape ? indices : compute_indices(index, nd_shape.length, false, nd_shape, aux_indices));
        }
        else
        {
            return is_value ? data : data[index];
        }
    }
    function set(index, indices, value)
    {
        if (refs)
        {
            refs[0].set(indices, value);
        }
        else if (stack)
        {
            for (var i=0; i<ndim; ++i)
            {
                aux_indices[i] = indices[i];
            }
            for (var i=0,sl=stack.length,t,tl; i<sl; ++i)
            {
                t = stack[i]; tl = t.shape(stack_axis);
                if (0 <= aux_indices[stack_axis] && aux_indices[stack_axis] < tl)
                {
                    t.set(aux_indices, value);
                    return;
                }
                aux_indices[stack_axis] -= tl;
            }
        }
        else if (nd_shape)
        {
            walk(data, same_shape ? indices : compute_indices(index, nd_shape.length, false, nd_shape, aux_indices), value);
        }
        else if (is_value)
        {
            data = value;
        }
        else
        {
            data[index] = value;
        }
    }

    o = o || {};

    is_transposed = _ ? !!_._transposed : false;

    if (data instanceof TensorView)
    {
        is_transposed = false;
        refs = [data];
        data = refs[0].data();
        total = refs[0].length();
        shape = o.shape || refs[0].shape();
        ndim = shape.length;
    }
    else if (_ && _._refs && (_._refs[0] instanceof TensorView))
    {
        is_transposed = false;
        refs = _._refs;
        total = refs[0].length();
        shape = (o.shape) || refs[0].shape();
        ndim = shape.length;
        if (refs[1])
        {
            data = null;
        }
    }
    else if (_ && _._stack && (2 <= _._stack.length) && (_._stack[0] instanceof TensorView) && (_._stack[1] instanceof TensorView))
    {
        data = null;
        stack_axis = _._stack_axis || 0;
        stack = _._stack;
        shape = stack[0].shape();
        shape[stack_axis] = sum(stack.map(function(t) {return t.shape(stack_axis);}));
        total = sum(stack.map(function(t) {return t.length();}));
        ndim = shape.length;
        aux_indices = new Array(ndim);
    }
    else
    {
        if (is_array(data))
        {
            is_value = false;
            if (o.ndarray && o.ndarray.length)
            {
                nd_shape = o.ndarray;
                total = product(nd_shape);
            }
            else if (is_array(data[0]))
            {
                nd_shape = compute_shape(data);
                total = product(nd_shape);
            }
            else
            {
                total = data.length;
            }
        }
        else
        {
            is_value = true;
            total = 1;
        }
        shape = o.shape;
        if (!shape || !shape.length) shape = [total];
        var computed_total = product(shape);
        if (is_value) total = computed_total;
        if (computed_total !== total) throw "TensorView:shape ["+shape.join(',')+"] does not match size "+String(total);
        ndim = shape.length;
        if (nd_shape)
        {
            same_shape = (nd_shape.length === shape.length) && (nd_shape.length === shape.filter(function(shapei, i) {return shapei === nd_shape[i];}).length);
            if (!same_shape) aux_indices = new Array(nd_shape.length);
        }
    }

    stride = new Array(ndim);
    if (is_transposed)
    {
        stride[0] = 1;
        for (i=1; i<ndim; ++i) stride[i] = stride[i-1]*shape[i-1];
    }
    else
    {
        stride[ndim-1] = 1;
        for (i=ndim-2; i>=0; --i) stride[i] = stride[i+1]*shape[i+1];
    }

    length = ndim ? product(shape) : 0;
    o = _ = null;

    self.dispose = function() {
        refs = null;
        stack = null;
        aux_indices = null;
        nd_shape = null;
        data = null;
        shape = null;
        stride = null;
    };
    self.clone = function() {
        return new TensorView(data, {
            ndarray: nd_shape,
            shape: shape.slice()
        }, {
            _transposed: is_transposed,
            _refs: refs,
            _stack: stack,
            _stack_axis: stack_axis
        });
    };
    self.data = function() {
        return data;
    };
    self.dimension = function() {
        return ndim;
    };
    self.shape = function(axis) {
        return arguments.length ? shape[axis] : shape.slice();
    };
    self.length = function() {
        return length;
    };
    self.index = function(/*indices*/) {
        return compute_index(Array.isArray(arguments[0]) ? arguments[0] : arguments, ndim, is_transposed, shape, stride);
    };
    self.indices = function(index) {
        if (0 > index || index >= total) throw "TensorView::indices:index ("+index+") is out of bounds (0,"+(total-1)+")!";
        return compute_indices(index, ndim, is_transposed, shape, stride);
    };
    self.get = function(/*indices*/) {
        var indices = Array.isArray(arguments[0]) ? arguments[0] : arguments, index = 0;
        if (indices.length < ndim) throw "TensorView::get:indices do not match shape dimension!";
        if (!stack) index = compute_index(indices, ndim, is_transposed, shape, stride);
        if (0 > index || index >= total) throw "TensorView::get:index ("+index+") is out of bounds (0,"+(total-1)+")!";
        return get(index, indices);
    };
    self.set = function(/*indices, value*/) {
        var indices = arguments, count = arguments.length-1, index = 0;
        if (Array.isArray(arguments[0])) {indices = arguments[0]; count = indices.length;}
        if (count < ndim) throw "TensorView::set:indices do not match shape dimension!";
        if (!stack) index = compute_index(indices, ndim, is_transposed, shape, stride);
        if (0 > index || index >= total) throw "TensorView::set:index ("+index+") is out of bounds (0,"+(total-1)+")!";
        set(index, indices, arguments[arguments.length-1]);
        return self;
    };
    self.iterator = function(order) {
        var i = 0 < length ? ("column-major" === order ? (0) : (ndim-1)) : -1,
            indices = null, ind = null, index = 0,
            value = [null, null], ret = {value: null};
        return {next:function next() {
            if ((0 > i) || (i >= ndim))
            {
                indices = ind = ret = value = null;
                return {done: true};
            }
            else
            {
                if (!indices)
                {
                    indices = (new Array(ndim)).fill(0);
                    ind = indices.slice();
                    index = 0;
                    value[0] = get(index, indices);
                    value[1] = ind;
                    ret.value = value;
                }
                else
                {
                    if ("column-major" === order)
                    {
                        // column-major
                        while ((i < ndim) && (indices[i]+1 >= shape[i]))
                        {
                            index -= indices[i] * stride[i];
                            ++i;
                        }
                        if (i < ndim)
                        {
                            indices[i] += 1;
                            ind[i] = indices[i];
                            index += stride[i];
                            while (0 <= i-1)
                            {
                                --i;
                                indices[i] = 0;
                                ind[i] = 0;
                            }
                            value[0] = get(index, indices);
                            value[1] = ind;
                            ret.value = value;
                        }
                        else
                        {
                            indices = ind = ret = value = null;
                            return {done: true};
                        }
                    }
                    else
                    {
                        // row-major
                        while ((i >= 0) && (indices[i]+1 >= shape[i]))
                        {
                            index -= indices[i] * stride[i];
                            --i;
                        }
                        if (0 <= i)
                        {
                            indices[i] += 1;
                            ind[i] = indices[i];
                            index += stride[i];
                            while (i+1 < ndim)
                            {
                                ++i;
                                indices[i] = 0;
                                ind[i] = 0;
                            }
                            value[0] = get(index, indices);
                            value[1] = ind;
                            ret.value = value;
                        }
                        else
                        {
                            indices = ind = ret = value = null;
                            return {done: true};
                        }
                    }
                }
                return ret;
            }
        }};
    };
    self.forEach = function(f, order) {
        if (0 < length && is_function(f))
        {
            var iter = self.iterator(order || "row-major"), next, ret = null;
            while (true)
            {
                next = iter.next();
                if (!next || next.done) return;
                ret = f(next.value[0], next.value[1], data, self);
                if (false === ret) return; // if false returned end forEach
            }
        }
    };
    self.transpose = function() {
        return !stack && !refs ? (new TensorView(data, {
            ndarray: nd_shape,
            shape: shape.slice().reverse()
            }, {
                _transposed: !is_transposed
            })) : (new TensorView(self, {
                shape: shape.slice().reverse()
            }, {
            _transposed: !is_transposed,
            _refs: refs ? refs.map(function(t) {return t.transpose();}).reverse() : null,
            _stack: stack ? stack.map(function(t) {return t.transpose();}) : null,
            _stack_axis: stack ? ndim-1-stack_axis : -1
        }));
    };
    self.reshape = function(shape) {
        return !stack && !refs ? new TensorView(data, {ndarray: nd_shape, shape: shape}, {_transposed: is_transposed}) : new TensorView(self, {shape: shape});
    };
    self.slice = function(/*slices*/) {
        var slices = compute_slices(Array.isArray(arguments[0]) ? arguments[0] : ([].slice.call(arguments)), shape),
            sliced_shape = slices.map(function(slice, i) {return compute_slice_count(shape[i], slice.start, slice.end, slice.step);}),
            //new_total = product(sliced_shape),
            new_shape = sliced_shape.filter(function(dim) {return 1 !== dim;}),
            new_ndim = new_shape.length,
            slice = self.clone(),
            _get = slice.get,
            _set = slice.set
        ;
        slice.get = function(/*indices*/) {
            var indices = Array.isArray(arguments[0]) ? arguments[0] : arguments, i = 0;
            if (indices.length < new_ndim) throw "TensorView::get:indices do not match shape dimension!";
            return _get.call(self, sliced_shape.map(function(dim, axis) {
                return 1 === dim ? slices[axis].start : (slices[axis].start + indices[i++] * slices[axis].step);
            }));
        };
        slice.set = function(/*indices, value*/) {
            var indices = arguments, count = arguments.length-1, i = 0;
            if (Array.isArray(arguments[0])) {indices = arguments[0]; count = indices.length;}
            if (count < new_ndim) throw "TensorView::set:indices do not match shape dimension!";
            _set.call(self, sliced_shape.map(function(dim, axis) {
                return 1 === dim ? slices[axis].start : (slices[axis].start + indices[i++] * slices[axis].step);
            }), arguments[arguments.length-1]);
            return slice;
        };
        return new TensorView(slice, {shape: new_shape}, {_transposed: is_transposed});
    };
    self.concat = function(others, axis) {
        axis = axis || 0;
        if (others instanceof TensorView) others = [others];
        for (var i=0,n=others.length; i<n; ++i)
        {
            var matchSize = shape.filter(function(shapej, j) {return j === axis || shapej === others[i].shape(j);});
            if (matchSize.length !== shape.length) throw "TensorView::concat:["+shape.map(function(shape, i) {return axis === i ? ':' : shape;}).join(',')+"] and ["+others[i].shape().map(function(shape, i) {return axis === i? ':' : shape;}).join(',')+"] shapes do not match!";
        }
        return new TensorView(
        null,
        null,
        {
            _stack: [self].concat(others),
            _stack_axis: axis
        }
        );
    };
    self.toArray = function(ArrayClass, order) {
        if ("string" === typeof ArrayClass)
        {
            order = ArrayClass;
            ArrayClass = Array;
        }
        var array = new (ArrayClass || Array)(length), index = 0;
        self.forEach(function(di/*,i*/) {
            // put in row-major or column-major order
            array[index++] = di;
        }, order || "row-major");
        return array;
    };
    self.toNDArray = function(order) {
        var ndarray = ndim ? new Array(shape["column-major" === order ? ndim-1 : 0]) : [];
        self.forEach("column-major" === order ? function(di, i) {
            // put in column-major order
            for (var a=ndarray,n=ndim-1,j=n,ij; j>0; --j)
            {
                ij = i[j];
                if (null == a[ij]) a[ij] = new Array(shape[j-1]);
                a = a[ij];
            }
            a[i[0]] = di;
        } : function(di, i) {
            // put in row-major order
            for (var a=ndarray,n=ndim-1,j=0,ij; j<n; ++j)
            {
                ij = i[j];
                if (null == a[ij]) a[ij] = new Array(shape[j+1]);
                a = a[ij];
            }
            a[i[n]] = di;
        }, order || "row-major");
        return ndarray;
    };
    self.toString = function(maxsize) {
        if (null == maxsize) maxsize = Infinity;
        var ndarray = self.toNDArray();
        return 2 < ndim ? str_nd(ndarray, maxsize) : (2 === ndim ? str_2d(ndarray, maxsize) : str_1d(ndarray, maxsize));
    };
}
TensorView.VERSION = '2.0.0';
TensorView[proto] = {
    constructor: TensorView,
    dispose: null,
    clone: null,
    data: null,
    dimension: null,
    shape: null,
    length: null,
    index: null,
    indices: null,
    get: null,
    set: null,
    iterator: null,
    forEach: null,
    transpose: null,
    reshape: null,
    slice: null,
    concat: null,
    toArray: null,
    toNDArray: null,
    toString: null
};
if (('undefined' !== typeof Symbol) && ('undefined' !== typeof Symbol.iterator))
{
    TensorView[proto][Symbol.iterator] = function() {
        return this.iterator();
    };
}

// utils
function is_function(x)
{
    return "function" === typeof x;
}
function is_array(x)
{
    if (Array.isArray(x)) return true;
    return TypedArray ? (x instanceof TypedArray) : false;
}
function array(n, v)
{
    n = stdMath.max(0, stdMath.round(n));
    var i, arr = new Array(n);
    for (i=0; i<n; ++i) arr[i] = is_function(v) ? v(i, arr) : v;
    return arr;
}
function walk(a, i, v)
{
    var ai = a, n = i.length-1, j;
    for (j=0; j<n; ++j) ai = ai[i[j]];
    if (2 < arguments.length) ai[i[n]] = v;
    return ai[i[n]];
}
function compute_shape(x)
{
    return is_array(x) ? ([x.length]).concat(compute_shape(x[0])) : [];
}
function compute_index(indices, ndim, transposed, shape, stride, slice)
{
    // compute single index for row/column-major ordering scheme from multidimensional indices
    var index = 0, axis, i;
    if (slice)
    {
        for (axis=0; axis<ndim; ++axis)
        {
            i = indices[axis];
            if (0 > i) i += shape[axis];
            if (0 > i || i >= shape[axis]) throw "TensorView:index ("+indices[axis]+") for dimension ("+axis+") is out of bounds (0,"+(shape[axis]-1)+")!";
            index += stride[axis] * (slice[axis].start + i * slice[axis].step);
        }
    }
    else
    {
        for (axis=0; axis<ndim; ++axis)
        {
            i = indices[axis];
            if (0 > i) i += shape[axis];
            if (0 > i || i >= shape[axis]) throw "TensorView:index ("+indices[axis]+") for dimension ("+axis+") is out of bounds (0,"+(shape[axis]-1)+")!";
            index += stride[axis] * i;
        }
    }
    return index;
}
function compute_indices(index, ndim, transposed, shape, indices, slice)
{
    // compute multidimensional indices for row/column-major ordering scheme from single index
    indices = indices || new Array(ndim);
    var axis, i;
    if (transposed)
    {
        if (slice)
        {
            for (axis=0; axis<ndim; ++axis)
            {
                i = index % shape[axis];
                index = stdMath.floor(index / shape[axis]);
                indices[axis] = stdMath.floor((i - slice[axis].start) / slice[axis].step);
            }
        }
        else
        {
            for (axis=0; axis<ndim; ++axis)
            {
                i = index % shape[axis];
                index = stdMath.floor(index / shape[axis]);
                indices[axis] = i;
            }
        }
    }
    else
    {
        if (slice)
        {
            for (axis=ndim-1; axis>=0; --axis)
            {
                i = index % shape[axis];
                index = stdMath.floor(index / shape[axis]);
                indices[axis] = stdMath.floor((i - slice[axis].start) / slice[axis].step);
            }
        }
        else
        {
            for (axis=ndim-1; axis>=0; --axis)
            {
                i = index % shape[axis];
                index = stdMath.floor(index / shape[axis]);
                indices[axis] = i;
            }
        }
    }
    return indices;
}
function compute_slices(slices, shape)
{
    while (slices.length < shape.length) slices = slices.concat(":");
    return slices.map(function(slice, i) {
        if (":" === slice)
        {
            slice = {start:0, end:shape[i]-1, step:1};
        }
        else if ("string" === typeof slice)
        {
            if (-1 < slice.indexOf(":"))
            {
                slice = slice.split(":");
                slice = slice.length > 2 ? {start:+slice[0], end:+slice[2], step:+slice[1]} : {start:+slice[0], end:+slice[1], step:1};
            }
            else
            {
                slice = parseInt(slice);
                slice = {start:slice, end:slice, step:1};
            }
        }
        else if (Array.isArray(slice))
        {
            slice = slice.length > 2 ? {start:slice[0], end:slice[2], step:slice[1]} : {start:slice[0], end:slice[1], step:1};
        }
        else if ("number" === typeof slice)
        {
            slice = {start:slice, end:slice, step:1};
        }
        else
        {
            slice = {start:0, end:shape[i]-1, step:1};
        }
        if (null == slice.start) slice.start = 0;
        if (null == slice.end) slice.end = shape[i]-1;
        if (null == slice.step) slice.step = 1;
        slice.start = slice.start || 0;
        slice.end = slice.end || 0;
        slice.step = slice.step || 1;
        if (0 > slice.start) slice.start += shape[i];
        if (0 > slice.end) slice.end += shape[i];
        slice.start = clamp(slice.start, 0, shape[i]-1);
        slice.end = clamp(slice.end, 0, shape[i]-1);
        slice.step = clamp(slice.step, -shape[i], shape[i]);
        slice.end = slice.start + slice.step*stdMath.floor(stdMath.abs(slice.end-slice.start)/stdMath.abs(slice.step));
        return slice;
    });
}
function compute_slice_count(length, start, end, step)
{
    if (!length || (0 > step && (start < 0 || start < end)) || (0 < step && (start >= length || start > end))) return 0;
    return stdMath.min(length, stdMath.ceil((stdMath.abs(end-start)+1)/stdMath.abs(step)));
}
function pad(s, n, z, after)
{
    var p = s.length < n ? (new Array(n-s.length+1)).join(z) : '';
    return after ? (s + p) : (p + s);
}
function add(a, b)
{
    return a + b;
}
function mul(a, b)
{
    return a * b;
}
function sum(array)
{
    return array.reduce(add, 0);
}
function product(array)
{
    return array.reduce(mul, 1);
}
function clamp(x, min, max)
{
    return stdMath.min(stdMath.max(x, min), max);
}
function str_1d(x, MAXPRINTSIZE)
{
    if (x.length > MAXPRINTSIZE)
    {
        x = x.slice(0, stdMath.round(MAXPRINTSIZE/2)).concat(['..']).concat(x.slice(-stdMath.round(MAXPRINTSIZE/2)+1));
    }
    return '[' + x.map(function(xi) {return String(xi);}).join('  ') + ']';
}
function str_2d(x, MAXPRINTSIZE)
{
    var use_ddots = false;
    if (x[0].length > MAXPRINTSIZE)
    {
        x = x.map(function(row) {
            return row.slice(0, stdMath.round(MAXPRINTSIZE/2)).concat('..').concat(row.slice(-stdMath.round(MAXPRINTSIZE/2)+1));
        });
        use_ddots = true;
    }
    if (x.length > MAXPRINTSIZE)
    {
        x = x.slice(0, stdMath.round(MAXPRINTSIZE/2)).concat([array(x[0].length, function(i) {return stdMath.round(MAXPRINTSIZE/2) === i ? (use_ddots ? ':.' : ':') : ':';})]).concat(x.slice(-stdMath.round(MAXPRINTSIZE/2)+1));
    }
    var ln = array(x[0].length, function(col) {
        return x.map(function(row) {return row[col];}).reduce(function(l, xi) {
            return stdMath.max(l, String(xi).length);
        }, 0);
    });
    return x.map(function(row, i) {
        return '[' + row.map(function(xij, j) {
            return pad(String(xij), ln[j], ' ');
        }).join('  ') + ']';
    }).join('\n');
}
function str_nd(x, MAXPRINTSIZE, indices)
{
    if (null == indices) indices = [];
    var str = '', i, n = x.length, lim = stdMath.min(n, stdMath.round(MAXPRINTSIZE/2));
    for (i=0; i<lim; ++i)
    {
        if (is_array(x[i]) && is_array(x[i][0]))
        {
            if (is_array(x[i][0][0]))
            {
                if (str.length) str += "\n";
                str += str_nd(x[i], MAXPRINTSIZE, indices.concat(i));
            }
            else
            {
                if (str.length) str += "\n";
                str += 'array(' + indices.concat([i, ':', ':']).map(String).join(',') + ') ->' + "\n" + str_2d(x[i], MAXPRINTSIZE);
            }
        }
    }
    if (lim < n)
    {
        if (str.length) str += "\n" + indices.concat(array(compute_shape(x).length, function() {return ':';})).map(String).join(' ');
        for (i=n-lim; i<n; ++i)
        {
            if (is_array(x[i]) && is_array(x[i][0]))
            {
                if (is_array(x[i][0][0]))
                {
                    if (str.length) str += "\n";
                    str += str_nd(x[i], MAXPRINTSIZE, indices.concat(i));
                }
                else
                {
                    if (str.length) str += "\n";
                    str += 'array(' + indices.concat([i, ':', ':']).map(String).join(',') + ') ->' + "\n" + str_2d(x[i], MAXPRINTSIZE);
                }
            }
        }
    }
    return str;
}

// export it
return TensorView;
});
