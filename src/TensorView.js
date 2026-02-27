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
    def = Object.defineProperty,
    NOP = function() {},
    TypedArray = "undefined" !== typeof Float32Array ? Object.getPrototypeOf(Float32Array) : null;

function TensorView(data, o, _)
{
    if (!(this instanceof TensorView)) return new TensorView(data, o, _);

    var self = this,
        nd_shape = null,
        shape = null,
        ndim = 0,
        stride = null,
        ref = null,
        is_transposed = false,
        is_value = false,
        same_shape = false,
        length = 0,
        total = 0,
        computed_total = null,
        aux_indices = null,
        axis, index_getter,
        indices_getter,
        getter, setter;

    function get(index, indices)
    {
        if (getter)
        {
            return getter(indices);
        }
        else if (ref)
        {
            return ref.get(ref.indices(index));
        }
        else if (nd_shape)
        {
            return walk(data, same_shape ? indices : compute_indices(indices_getter, index, nd_shape.length, false, nd_shape, aux_indices));
        }
        else
        {
            return is_value ? data : data[index];
        }
    }
    function set(index, indices, value)
    {
        if (setter)
        {
            setter(indices, value);
        }
        else if (ref)
        {
            ref.set(ref.indices(index), value);
        }
        else if (nd_shape)
        {
            walk(data, same_shape ? indices : compute_indices(indices_getter, index, nd_shape.length, false, nd_shape, aux_indices), value);
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

    is_transposed = _ ? !!_.transposed : false;
    index_getter = _ && is_function(_.index) ? _.index : null;
    indices_getter = _ && is_function(_.indices) ? _.indices : null;
    getter = _ && is_function(_.get) ? _.get : null;
    setter = _ && is_function(_.set) ? _.set : null;
    shape = o.shape;

    if (data instanceof TensorView)
    {
        ref = data;
        data = ref.data;
        shape = shape || ref.shape();
        //total = ref.length;
        computed_total = total = product(shape);
    }
    else if (is_array(data))
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
        if (!shape || !shape.length) shape = [total];
        computed_total = product(shape);
    }
    else if (null == data)
    {
        is_value = false;
        computed_total = total = product(shape);
    }
    else
    {
        is_value = true;
        computed_total = total = product(shape);
    }

    if (computed_total !== total) throw "TensorView:shape ["+shape.join(',')+"] does not match size "+String(total);
    ndim = shape.length;
    if (nd_shape)
    {
        same_shape = (nd_shape.length === shape.length) && (nd_shape.length === shape.filter(function(dim, axis) {return dim === nd_shape[axis];}).length);
        if (!same_shape) aux_indices = new Array(nd_shape.length);
    }

    if (Array.isArray(o.stride) && (o.stride.length === shape.length))
    {
        stride = o.stride;
    }
    else
    {
        stride = new Array(ndim);
        if (is_transposed)
        {
            stride[0] = 1;
            for (axis=1; axis<ndim; ++axis) stride[axis] = stride[axis-1] * shape[axis-1];
        }
        else
        {
            stride[ndim-1] = 1;
            for (axis=ndim-2; axis>=0; --axis) stride[axis] = stride[axis+1] * shape[axis+1];
        }
    }

    length = ndim ? total : 0;
    o = _ = null;

    self.dispose = function() {
        ref = null;
        data = null;
        shape = null;
        stride = null;
        nd_shape = null;
        index_getter = null;
        indices_getter = null;
        getter = null;
        setter = null;
        aux_indices = null;
    };
    def(self, 'data', {
        get: function() {return data;},
        set: NOP,
        enumerable: true,
        configurable: false
    });
    def(self, 'dimension', {
        get: function() {return ndim;},
        set: NOP,
        enumerable: true,
        configurable: false
    });
    def(self, 'length', {
        get: function() {return length;},
        set: NOP,
        enumerable: true,
        configurable: false
    });
    self.shape = function(axis) {
        return true === axis ? shape : (arguments.length ? shape[axis] : shape.slice());
    };
    self.index = function(/*indices*/) {
        return compute_index(index_getter, Array.isArray(arguments[0]) ? arguments[0] : arguments, ndim, is_transposed, shape, stride);
    };
    self.indices = function(index) {
        index = index || 0;
        if (0 > index) index += total;
        if (0 > index || index >= total) throw "TensorView::indices:index ("+index+") is out of bounds (0,"+(total-1)+")!";
        return compute_indices(indices_getter, index, ndim, is_transposed, shape);
    };
    self.get = function(/*indices*/) {
        var indices = Array.isArray(arguments[0]) ? arguments[0] : arguments, index = 0;
        if (indices.length < ndim) throw "TensorView::get:indices do not match shape dimension!";
        index = compute_index(index_getter, indices, ndim, is_transposed, shape, stride);
        if (0 > index || index >= total) throw "TensorView::get:index ("+index+") is out of bounds (0,"+(total-1)+")!";
        return get(index, indices);
    };
    self.set = function(/*indices, value*/) {
        var indices = arguments, count = arguments.length-1, index = 0;
        if (Array.isArray(arguments[0])) {indices = arguments[0]; count = indices.length;}
        if (count < ndim) throw "TensorView::set:indices do not match shape dimension!";
        index = compute_index(index_getter, indices, ndim, is_transposed, shape, stride);
        if (0 > index || index >= total) throw "TensorView::set:index ("+index+") is out of bounds (0,"+(total-1)+")!";
        set(index, indices, arguments[arguments.length-1]);
        return self;
    };
    self.iterator = function(order) {
        var axis = 0 < length ? ("column-major" === order ? (0) : (ndim-1)) : -1,
            indices = null, ind = null, index = 0,
            value = [null, null, null], ret = {value: null};
        return {next:function next() {
            if ((0 > axis) || (axis >= ndim))
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
                    value[2] = index;
                    ret.value = value;
                }
                else
                {
                    if ("column-major" === order)
                    {
                        // column-major
                        while ((axis < ndim) && (indices[axis]+1 >= shape[axis]))
                        {
                            index -= indices[axis] * stride[axis];
                            ++axis;
                        }
                        if (axis < ndim)
                        {
                            indices[axis] += 1;
                            ind[axis] = indices[axis];
                            index += stride[axis];
                            while (0 <= axis-1)
                            {
                                --axis;
                                indices[axis] = 0;
                                ind[axis] = 0;
                            }
                            value[0] = get(index, indices);
                            value[1] = ind;
                            value[2] = index;
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
                        while ((axis >= 0) && (indices[axis]+1 >= shape[axis]))
                        {
                            index -= indices[axis] * stride[axis];
                            --axis;
                        }
                        if (0 <= axis)
                        {
                            indices[axis] += 1;
                            ind[axis] = indices[axis];
                            index += stride[axis];
                            while (axis+1 < ndim)
                            {
                                ++axis;
                                indices[axis] = 0;
                                ind[axis] = 0;
                            }
                            value[0] = get(index, indices);
                            value[1] = ind;
                            value[2] = index;
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
            for (;;)
            {
                next = iter.next();
                if (!next || next.done) return;
                ret = f(next.value[0], next.value[1], data, self);
                if (false === ret) return; // if false returned end forEach
            }
        }
    };
    self.transpose = function() {
        return ref && is_transposed ? ref /*idempotent*/ : new TensorView(self, {shape: shape.slice().reverse()}, {transposed: !is_transposed});
    };
    self.reshape = function(new_shape) {
        if (ref)
        {
            if (shape.length === shape.filter(function(dim, axis) {return dim === new_shape[axis];}).length)
            {
                return ref; /*idempotent*/
            }
        }
        else if (data)
        {
            return new TensorView(data, {shape: new_shape}); /*trivial reshape*/
        }
        return new TensorView(self, {shape: new_shape});
    };
    self.squeeze = function() {
        var new_shape = shape.filter(function(dim) {return 1 !== dim;}), new_ndim,
            adjust_indices = function(indices) {
                var i = 0;
                return shape.map(function(dim, axis) {
                    return 1 === dim ? 0 : indices[i++];
                });
            };
        if (!new_shape.length) new_shape = [1];
        new_ndim = new_shape.length;
        return new_ndim === ndim ? self /*idempotent*/ : new TensorView(self, {
            shape: new_shape
        }, {
            index: function(indices) {
                return self.index(adjust_indices(indices));
            },
            indices: function(index) {
                return self.indices(index).reduce(function(indices, i, axis) {
                    if (1 !== shape[axis]) indices.push(i);
                    return indices;
                }, (1 === new_dim) && (1 === new_shape[0]) ? [0] : []);
            },
            get: function(indices) {
                if (indices.length < new_ndim) throw "TensorView::get:indices do not match shape dimension!";
                return self.get(adjust_indices(indices));
            },
            set: function(indices, value) {
                if (indices.length < new_ndim) throw "TensorView::set:indices do not match shape dimension!";
                self.set(adjust_indices(indices), value);
            }
        });
    };
    self.slice = function(/*slices*/) {
        var slices = compute_slices(Array.isArray(arguments[0]) ? arguments[0] : ([].slice.call(arguments)), shape),
            new_shape = slices.map(function(slice, axis) {
                return compute_slice_length(shape[axis], slice.start, slice.end, slice.step);
            }),
            adjust_indices = function(indices) {
                return slices.map(function(slice, axis) {
                    return slice.start + indices[axis] * slice.step;
                });
            }
        ;
        return shape.length === shape.filter(function(dim, axis) {return dim === new_shape[axis];}).length ? self /*idempotent*/ : new TensorView(self, {
            shape: new_shape
        }, {
            get: function(indices) {
                if (indices.length < ndim) throw "TensorView::get:indices do not match shape dimension!";
                return self.get(adjust_indices(indices));
            },
            set: function(indices, value) {
                if (indices.length < ndim) throw "TensorView::set:indices do not match shape dimension!";
                self.set(adjust_indices(indices), value);
            }
        });
    };
    self.concat = function(others, axis) {
        axis = axis || 0;
        if (others instanceof TensorView) others = [others];
        for (var i=0,n=others.length,matchShape; i<n; ++i)
        {
            matchShape = shape.filter(function(dim, a) {return (axis === a) || (dim === others[i].shape(a));});
            if (matchShape.length !== shape.length) throw "TensorView::concat:["+shape.map(function(dim, a) {return axis === a ? ':' : dim;}).join(',')+"] and ["+others[i].shape().map(function(dim, a) {return axis === a ? ':' : dim;}).join(',')+"] shapes do not match!";
        }
        var aux_indices = new Array(ndim), stack = [self].concat(others);
        return new TensorView(null, {
            shape: shape.map(function(dim, a) {
                return axis === a ? others.reduce(function(total, other) {
                    return total + other.shape(axis);
                }, dim) : dim;
            })
        }, {
            get: function(indices) {
                for (var i=0; i<ndim; ++i)
                {
                    aux_indices[i] = indices[i];
                }
                for (var i=0,n=stack.length,t,tl; i<n; ++i)
                {
                    t = stack[i]; tl = t.shape(axis);
                    if (0 <= aux_indices[axis] && aux_indices[axis] < tl) return t.get(aux_indices);
                    aux_indices[axis] -= tl;
                }
            },
            set: function(indices, value) {
                for (var i=0; i<ndim; ++i)
                {
                    aux_indices[i] = indices[i];
                }
                for (var i=0,n=stack.length,t,tl; i<n; ++i)
                {
                    t = stack[i]; tl = t.shape(axis);
                    if (0 <= aux_indices[axis] && aux_indices[axis] < tl)
                    {
                        t.set(aux_indices, value);
                        return;
                    }
                    aux_indices[axis] -= tl;
                }
            }
        });
    };
}
TensorView.VERSION = '2.0.0';
TensorView[proto] = {
    constructor: TensorView,
    dispose: null,
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
    squeeze: null,
    slice: null,
    concat: null,
    toArray: function(ArrayClass, order) {
        if ("string" === typeof ArrayClass)
        {
            order = ArrayClass;
            ArrayClass = Array;
        }
        var self = this, array = new (ArrayClass || Array)(self.length), index = 0;
        self.forEach(function(di/*,i*/) {
            // put in row-major or column-major order
            array[index++] = di;
        }, order || "row-major");
        return array;
    },
    toNDArray: function(order) {
        var self = this, shape = self.shape(true), ndim = shape.length,
            ndarray = ndim ? new Array(shape["column-major" === order ? ndim-1 : 0]) : [];
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
    },
    toString: function(maxsize, stringify) {
        if (!is_function(stringify)) stringify = function(x) {return String(x);};
        if ('number' !== typeof maxsize) maxsize = Infinity;
        var self = this, shape = self.shape(), ndim = shape.length, ndarray = self.toNDArray();
        return 2 < ndim ? str_nd(ndarray, shape, maxsize, stringify) : (2 === ndim ? str_2d(ndarray, shape, maxsize, stringify) : str_1d(ndarray, shape, maxsize, stringify));
    }
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
function compute_index(index_getter, indices, ndim, transposed, shape, stride)
{
    // compute single index for row/column-major ordering scheme from multidimensional indices
    if (index_getter) return index_getter(indices);
    var index = 0, axis, i;
    for (axis=0; axis<ndim; ++axis)
    {
        i = indices[axis];
        if (0 > i) i += shape[axis];
        if (0 > i || i >= shape[axis]) throw "TensorView:index ("+indices[axis]+") for dimension ("+axis+") is out of bounds (0,"+(shape[axis]-1)+")!";
        index += stride[axis] * i;
    }
    return index;
}
function compute_indices(indices_getter, index, ndim, transposed, shape, indices)
{
    // compute multidimensional indices for row/column-major ordering scheme from single index
    if (indices_getter) return indices_getter(index);
    indices = indices || new Array(ndim);
    var axis, i;
    if (transposed)
    {
        for (axis=0; axis<ndim; ++axis)
        {
            i = index % shape[axis];
            index = stdMath.floor(index / shape[axis]);
            indices[axis] = i;
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
function compute_slice_length(length, start, end, step)
{
    if (!length || (0 > step && (start < 0 || start < end)) || (0 < step && (start >= length || start > end))) return 0;
    return stdMath.min(length, stdMath.ceil((stdMath.abs(end-start)+1)/stdMath.abs(step)));
}
function pad(s, n, z, after)
{
    var p = s.length < n ? (new Array(n-s.length+1)).join(z) : '';
    return after ? (s + p) : (p + s);
}
function clamp(x, min, max)
{
    return stdMath.min(stdMath.max(x, min), max);
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
function project(x, i, j)
{
    j = j || 0;
    if (is_array(x))
    {
        return ':' === i[j] ? x.map(function(xj) {return project(xj, i, j+1);}) : project(x[i[j]], i, j+1);
    }
    return x;
}
function str_1d(x, shape, MAXPRINTSIZE, stringify)
{
    if (shape[0] > MAXPRINTSIZE)
    {
        x = x.slice(0, stdMath.round(MAXPRINTSIZE/2)).concat(['..']).concat(x.slice(-stdMath.round(MAXPRINTSIZE/2)+1));
    }
    return '[' + x.map(stringify).join('  ') + ']';
}
function str_2d(x, shape, MAXPRINTSIZE, stringify)
{
    var use_ddots = false;
    if (shape[1] > MAXPRINTSIZE)
    {
        x = x.map(function(row) {
            return row.slice(0, stdMath.round(MAXPRINTSIZE/2)).concat('..').concat(row.slice(-stdMath.round(MAXPRINTSIZE/2)+1));
        });
        use_ddots = true;
    }
    if (shape[0] > MAXPRINTSIZE)
    {
        x = x.slice(0, stdMath.round(MAXPRINTSIZE/2)).concat([array(x[0].length, function(i) {return stdMath.round(MAXPRINTSIZE/2) === i ? (use_ddots ? ':.' : ':') : ':';})]).concat(x.slice(-stdMath.round(MAXPRINTSIZE/2)+1));
    }
    var ln = array(x[0].length, function(col) {
        return x.map(function(row) {return row[col];}).reduce(function(l, xi) {
            return stdMath.max(l, stringify(xi).length);
        }, 0);
    });
    return x.map(function(row, i) {
        return '[' + row.map(function(xij, j) {
            return pad(stringify(xij), ln[j], ' ');
        }).join('  ') + ']';
    }).join('\n');
}
function str_nd(x, shape, MAXPRINTSIZE, stringify, indices)
{
    if (null == indices) indices = [];
    var str = '', ind, i, n, lim;
    if (shape.length === 2 + indices.length)
    {
        ind = [':', ':'].concat(indices);
        str += 'array(' + ind.map(String).join(',') + ') ->' + "\n" + str_2d(project(x, ind), shape, MAXPRINTSIZE, stringify);
    }
    else
    {
        n = shape[2+indices.length];
        lim = stdMath.min(n, stdMath.round(MAXPRINTSIZE/2));
        for (i=0; i<lim; ++i)
        {
            if (str.length) str += "\n";
            str += str_nd(x, shape, MAXPRINTSIZE, stringify, indices.concat(i));
        }
        if (lim < n)
        {
            if (str.length) str += "\n:";
            for (i=n-lim; i<n; ++i)
            {
                if (str.length) str += "\n";
                str += str_nd(x, shape, MAXPRINTSIZE, stringify, indices.concat(i));
            }
        }
    }
    return str;
}

// export it
return TensorView;
});
