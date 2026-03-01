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
        ordin = null,
        ordout = null,
        stride = null,
        ref = null,
        is_transposed = false,
        is_value = false,
        same_shape = false,
        length = 0,
        total = 0,
        computed_total = null,
        aux_indices = null,
        index_getter,
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
            return walk(data, same_shape ? indices : compute_indices(null, index, nd_shape.length, false, nd_shape, null, null, aux_indices));
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
            walk(data, same_shape ? indices : compute_indices(null, index, nd_shape.length, false, nd_shape, null, null, aux_indices), value);
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
    ordout = o.out_order;
    ordin = o.in_order || ordout;

    if (data instanceof TensorView)
    {
        ref = data;
        data = ref.data;
        shape = shape || ref.shape();
        ordout = ordout || ref.out_order();
        ordin = ordin || ref.in_order();
        //total = ref.length;
        computed_total = total = product(shape);
    }
    else if (is_array(data))
    {
        is_value = false;
        if (is_array(o.ndarray, true) && o.ndarray.length)
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
        if (!shape || !shape.length) shape = nd_shape ? nd_shape.slice() : [total];
        computed_total = product(shape);
    }
    else
    {
        is_value = null != data;
        computed_total = total = product(shape);
    }

    if (computed_total !== total) throw "TensorView shape ("+shape.join(',')+") does not match size "+String(total);
    ndim = shape.length;

    ordin = compute_order(ndim, ordin);
    ordout = compute_order(ndim, ordout);

    stride = compute_stride(ndim, shape, ordin, ordout, is_transposed);

    if (nd_shape)
    {
        same_shape = (nd_shape.length === shape.length) && (nd_shape.length === shape.filter(function(dim, axis) {return dim === nd_shape[axis];}).length) && (ordin.length === ordin.filter(function(axis, i) {return axis === i;}).length) && (ordout.length === ordout.filter(function(axis, i) {return axis === i;}).length);
        if (!same_shape) aux_indices = new Array(nd_shape.length);
    }

    length = ndim ? total : 0;
    o = _ = null;

    self.dispose = function() {
        ref = null;
        data = null;
        shape = null;
        ordin = null;
        ordout = null;
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
    self.in_order = function() {
        return true === arguments[0] ? ordin : (ordin.slice());
    };
    self.out_order = function() {
        return true === arguments[0] ? ordout : (ordout.slice());
    };
    self.stride = function(axis) {
        return true === axis ? stride : (arguments.length ? stride[axis] : stride.slice());
    };
    self.index = function(/*indices*/) {
        return compute_index(index_getter, is_array(arguments[0], true) ? arguments[0] : arguments, ndim, is_transposed, shape, ordin, ordout, stride);
    };
    self.indices = function(index) {
        index = index || 0;
        if (0 > index) index += total;
        if (0 > index || index >= total) throw "TensorView::indices index ("+index+") is out of bounds (0,"+(total-1)+")!";
        return compute_indices(indices_getter, index, ndim, is_transposed, shape, ordin, ordout);
    };
    self.get = function(/*indices*/) {
        var indices = is_array(arguments[0], true) ? arguments[0] : arguments, index = 0;
        if (indices.length < ndim) throw "TensorView::get indices do not match shape dimension!";
        index = compute_index(index_getter, indices, ndim, is_transposed, shape, ordin, ordout, stride);
        if (0 > index || index >= total) throw "TensorView::get index ("+index+") is out of bounds (0,"+(total-1)+")!";
        return get(index, indices);
    };
    self.set = function(/*indices, value*/) {
        var indices = arguments, count = arguments.length-1, index = 0;
        if (is_array(arguments[0], true)) {indices = arguments[0]; count = indices.length;}
        if (count < ndim) throw "TensorView::set indices do not match shape dimension!";
        index = compute_index(index_getter, indices, ndim, is_transposed, shape, ordin, ordout, stride);
        if (0 > index || index >= total) throw "TensorView::set index ("+index+") is out of bounds (0,"+(total-1)+")!";
        set(index, indices, arguments[arguments.length-1]);
        return self;
    };
    self.iterator = function() {
        var axis = 0 < length ? ndim-1 : -1, dax = -1, ax = axis,
            indices = null, ind = null, index = 0, indx = 0,
            value = [null, null, null], ret = {value: null};
        return {next:function next() {
            if ((0 > ax) || (ax >= ndim))
            {
                indices = ind = ret = value = null;
                return {done: true};
            }
            else
            {
                if (!indices)
                {
                    indices = array(ndim, 0);
                    index = 0;

                    ind = indices.slice();
                    indx = 0;

                    value[0] = get(index, indices);
                    value[1] = ind;
                    value[2] = indx;
                    ret.value = value;
                }
                else
                {
                    while ((0 <= ax && ax < ndim) && (ind[ordout[ax]]+1 >= shape[ordout[ax]]))
                    {
                        //indx -= ind[ordout[ax]] * stride[ordout[ax]];
                        ax += dax;
                    }

                    if (0 <= ax && ax < ndim)
                    {
                        ++ind[ordout[ax]];
                        ++indx;//indx += stride[ordout[ax]];
                        while (0 <= ax-dax && ax-dax < ndim)
                        {
                            ax -= dax;
                            ind[ordout[ax]] = 0;
                        }

                        while ((0 <= axis && axis < ndim) && (indices[ordin[axis]]+1 >= shape[ordin[axis]]))
                        {
                            index -= indices[ordin[axis]] * stride[ordin[axis]];
                            axis += dax;
                        }

                        ++indices[ordin[axis]];
                        index += stride[ordin[axis]];
                        while (0 <= axis-dax && axis-dax < ndim)
                        {
                            axis -= dax;
                            indices[ordin[axis]] = 0;
                        }

                        value[0] = get(index, indices);
                        value[1] = ind;
                        value[2] = indx;
                        ret.value = value;
                    }
                    else
                    {
                        indices = ind = ret = value = null;
                        return {done: true};
                    }
                }
                return ret;
            }
        }};
    };
    self.forEach = function(f, ordin) {
        if (0 < length && is_function(f))
        {
            var iter = self.iterator(), next, ret = null;
            for (;;)
            {
                next = iter.next();
                if (!next || next.done) return;
                ret = f(next.value[0], next.value[1]/*, next.value[2]*/, data, self);
                if (false === ret) return; // if false returned end forEach
            }
        }
    };
    self.transpose = function() {
        return ref && is_transposed ? ref /*idempotent*/ : new TensorView(self, {shape: shape.slice().reverse(), in_order: ordin.slice(), out_order: ordout.slice()}, {transposed: !is_transposed});
    };
    self.reorder = function(new_in_order, new_out_order) {
        new_in_order = compute_order(ndim, new_in_order || ordin.slice());
        new_out_order = compute_order(ndim, new_out_order || ordout.slice());
        if (new_in_order.length !== ndim || new_out_order.length !== ndim)
        {
            throw "TensorView::reorder ordin not valid or does not match shape dimension!";
        }
        return ordin.length === ordin.filter(function(axis, i) {return axis === new_in_order[i];}).length && ordout.length === ordout.filter(function(axis, i) {return axis === new_out_order[i];}).length ? self /*idempotent*/ : new TensorView(!ref && (null != data) ? data : self, {shape: shape.slice(), in_order: new_in_order, out_order: new_out_order});
    };
    self.reshape = function(new_shape) {
        var new_ndim = new_shape.length;
        if (
            (ndim === new_ndim) &&
            (new_ndim === shape.filter(function(dim, axis) {return dim === new_shape[axis];}).length)
        )
        {
            return self; /*idempotent*/
        }
        if (
            ref &&
            (ref.dimension === new_ndim) &&
            (new_ndim === ref.shape(true).filter(function(dim, axis) {return dim === new_shape[axis];}).length)
        )
        {
            return ref; /*idempotent*/
        }
        return new TensorView(!ref && (null != data) ? data : self, {shape: new_shape});
    };
    self.permute = function(/*permutation*/) {
        var permutation = is_array(arguments[0], true) ? arguments[0] : [].slice.call(arguments);
        if (!is_perm(permutation, ndim))
        {
            throw "TensorView::permute permutation not valid or does not match shape dimension!";
        }
        var new_shape = permutation.map(function(pi) {return shape[pi];}),
            ipermutation = invperm(permutation),
            adjust_indices = function(indices) {
                return ipermutation.map(function(pi, axis) {
                    return indices[pi];
                });
            };
        return permutation.length === permutation.filter(function(pi, i) {return pi === i;}).length ? self /*identity*/ : new TensorView(self, {
            shape: new_shape,
            in_order: ordin.slice(),
            out_order: ordout.slice()
        }, {
            get: function(indices) {
                if (indices.length < ndim) throw "TensorView::get indices do not match shape dimension!";
                return self.get(adjust_indices(indices));
            },
            set: function(indices, value) {
                if (indices.length < ndim) throw "TensorView::set indices do not match shape dimension!";
                self.set(adjust_indices(indices), value);
            }
        });
    };
    self.squeeze = function(start_axis) {
        start_axis = start_axis || 0;
        var new_shape = shape.filter(function(dim, axis) {
                return (axis < start_axis) || (1 !== dim);
            }),
            new_ndim,
            adjust_indices = function(indices) {
                var i = 0;
                return shape.map(function(dim, axis) {
                    return axis < start_axis ? indices[i++] : (1 === dim ? 0 : indices[i++]);
                });
            };
        if (!new_shape.length) new_shape = [1];
        new_ndim = new_shape.length;
        return new_ndim === ndim ? self /*idempotent*/ : new TensorView(self, {
            shape: new_shape,
            in_order: ordin.reduce(function(order, axis) {
                if (axis < start_axis)
                {
                    order.push(axis);
                }
                else if (1 !== shape[axis])
                {
                    for (var i=axis-1; i>=0; --i)
                    {
                        if (1 === shape[i]) --axis;
                    }
                    order.push(axis);
                }
                return order;
            }, []),
            out_order: ordout.reduce(function(order, axis) {
                if (axis < start_axis)
                {
                    order.push(axis);
                }
                else if (1 !== shape[axis])
                {
                    for (var i=axis-1; i>=0; --i)
                    {
                        if (1 === shape[i]) --axis;
                    }
                    order.push(axis);
                }
                return order;
            }, [])
        }, {
            index: function(indices) {
                return self.index(adjust_indices(indices));
            },
            indices: function(index) {
                var indices = self.indices(index).reduce(function(indices, i, axis) {
                    if ((axis < start_axis) || (1 !== shape[axis])) indices.push(i);
                    return indices;
                }, []);
                if (!indices.length) indices = [0];
                return indices;
            },
            get: function(indices) {
                if (indices.length < new_ndim) throw "TensorView::get indices do not match shape dimension!";
                return self.get(adjust_indices(indices));
            },
            set: function(indices, value) {
                if (indices.length < new_ndim) throw "TensorView::set indices do not match shape dimension!";
                self.set(adjust_indices(indices), value);
            }
        });
    };
    self.slice = function(/*slices*/) {
        var slices = compute_slices(is_array(arguments[0], true) ? arguments[0] : ([].slice.call(arguments)), shape),
            new_shape = slices.map(function(slice, axis) {
                return compute_slice_length(shape[axis], slice);
            }),
            adjust_indices = function(indices) {
                return slices.map(function(slice, axis) {
                    return null != slice.start ? (slice.start + indices[axis] * slice.step) : (slice[indices[axis]]);
                });
            }
        ;
        return shape.length === shape.filter(function(dim, axis) {return dim === new_shape[axis];}).length ? self /*idempotent*/ : new TensorView(self, {
            shape: new_shape,
            in_order: ordin.slice(),
            out_order: ordout.slice()
        }, {
            get: function(indices) {
                if (indices.length < ndim) throw "TensorView::get indices do not match shape dimension!";
                return self.get(adjust_indices(indices));
            },
            set: function(indices, value) {
                if (indices.length < ndim) throw "TensorView::set indices do not match shape dimension!";
                self.set(adjust_indices(indices), value);
            }
        });
    };
    self.concat = function(others, on_axis) {
        if (others instanceof TensorView) others = [others];
        on_axis = on_axis || 0;
        var stack = [self].concat(others),
            dims, min_dim, max_dim,
            adjust_shape, aux_indices;
        if ("newaxis" === on_axis)
        {
            dims = stack.map(function(view) {return view.dimension;});
            min_dim = stdMath.min.apply(stdMath, dims);
            max_dim = stdMath.max.apply(stdMath, dims);
            if (max_dim === min_dim) ++max_dim;
            on_axis = max_dim-1;
            adjust_shape = function(view) {
                return view.shape().concat(array(max_dim-view.dimension, 1));
            };
            return self.reshape(adjust_shape(self)).concat(others.map(function(other) {return other.reshape(adjust_shape(other));}), on_axis);
        }
        for (var i=0,n=others.length,matchShape; i<n; ++i)
        {
            matchShape = shape.filter(function(dim, axis) {return (on_axis === axis) || (dim === others[i].shape(axis));});
            if (matchShape.length !== shape.length) throw "TensorView::concat ("+shape.map(function(dim, axis) {return on_axis === axis ? ':' : dim;}).join(',')+") and ("+others[i].shape().map(function(dim, axis) {return on_axis === axis ? ':' : dim;}).join(',')+") shapes do not match!";
        }
        aux_indices = new Array(ndim);
        return new TensorView(null, {
            shape: shape.map(function(dim, axis) {
                return on_axis === axis ? others.reduce(function(total, other) {
                    return total + other.shape(on_axis);
                }, dim) : dim;
            })
        }, {
            get: function(indices) {
                if (indices.length < ndim) throw "TensorView::get indices do not match shape dimension!";
                for (var i=0; i<ndim; ++i)
                {
                    aux_indices[i] = indices[i];
                }
                for (var i=0,n=stack.length,t,tl; i<n; ++i)
                {
                    t = stack[i]; tl = t.shape(on_axis);
                    if (0 <= aux_indices[on_axis] && aux_indices[on_axis] < tl) return t.get(aux_indices);
                    aux_indices[on_axis] -= tl;
                }
            },
            set: function(indices, value) {
                if (indices.length < ndim) throw "TensorView::set indices do not match shape dimension!";
                for (var i=0; i<ndim; ++i)
                {
                    aux_indices[i] = indices[i];
                }
                for (var i=0,n=stack.length,t,tl; i<n; ++i)
                {
                    t = stack[i]; tl = t.shape(on_axis);
                    if (0 <= aux_indices[on_axis] && aux_indices[on_axis] < tl)
                    {
                        t.set(aux_indices, value);
                        return;
                    }
                    aux_indices[on_axis] -= tl;
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
    length: null,
    shape: null,
    in_order: null,
    out_order: null,
    stride: null,
    index: null,
    indices: null,
    get: null,
    set: null,
    iterator: null,
    forEach: null,
    transpose: null,
    reorder: null,
    reshape: null,
    permute: null,
    squeeze: null,
    slice: null,
    concat: null,
    toArray: function(ArrayClass) {
        var self = this, array = new (ArrayClass || Array)(self.length), index = 0;
        self.forEach(function(item/*, indices*/) {array[index++] = item;});
        return array;
    },
    toNDArray: function() {
        var self = this,
            shape = self.shape(true),
            ndim = shape.length,
            ndarray = ndim ? new Array(shape[0]) : [];
        self.forEach(function(item, indices) {
            for (var a=ndarray,n=ndim-1,i=0,ind; i<n; ++i)
            {
                ind = indices[i];
                if (null == a[ind]) a[ind] = new Array(shape[i+1]);
                a = a[ind];
            }
            a[indices[n]] = item;
        });
        return ndarray;
    },
    toString: function(maxsize, stringify) {
        if (!is_function(stringify)) stringify = function(item) {return String(item);};
        if (!is_num(maxsize, true)) maxsize = Infinity;
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
function is_obj(x)
{
    return "object" === typeof x;
}
function is_num(x, strict)
{
    return ("number" === typeof x) && (!strict || !isNaN(x));
}
function is_string(x)
{
    return "string" === typeof x;
}
function is_array(x, strict)
{
    if (Array.isArray(x)) return true;
    if (strict) return false;
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
function compute_stride(ndim, shape, ordin, ordout, is_transposed)
{
    var axis, stride = new Array(ndim);
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
    return stride;
}
function compute_order(ndim, order)
{
    if ("column-major" === order)
    {
        order = array(ndim, function(axis) {return (axis < 2 ? 1-axis : axis);});
    }
    else if ("row-major-reverse" === order)
    {
        order = array(ndim, function(axis) {return ndim-1-axis;});
    }
    else if (("row-major" === order) || !order || (ndim !== order.length))
    {
        order = array(ndim, function(axis) {return axis;});
    }
    return order;
}
function compute_index(index_getter, indices, ndim, is_transposed, shape, ordin, ordout, stride)
{
    // compute single index for arbitrary ordering scheme from multidimensional indices
    if (index_getter) return index_getter(indices);
    var index = 0, ax, axis, i;
    for (ax=0; ax<ndim; ++ax)
    {
        axis = ax;//ordin[ax];
        i = indices[axis];
        if (0 > i) i += shape[axis];
        if (0 > i || i >= shape[axis]) throw "TensorView index ("+indices[axis]+") for dimension ("+axis+") is out of bounds (0,"+(shape[axis]-1)+")!";
        index += stride[axis] * i;
    }
    return index;
}
function compute_indices(indices_getter, index, ndim, is_transposed, shape, ordin, ordout, indices)
{
    // compute multidimensional indices for arbitrary ordering scheme from single index
    if (indices_getter) return indices_getter(index);
    indices = indices || new Array(ndim);
    var ax, axis, i;
    if (is_transposed)
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
    return slices.slice(0, shape.length).map(function(slice, i) {
        if (is_string(slice))
        {
            slice = slice.trim();
            if (":" === slice)
            {
                slice = {start:0, end:shape[i]-1, step:1};
            }
            else if (-1 < slice.indexOf(":"))
            {
                slice = slice.split(":");
                slice = slice.length > 2 ? {start:+slice[0], end:+slice[2], step:+slice[1]} : {start:+slice[0], end:+slice[1], step:1};
            }
            else if (-1 < slice.indexOf(","))
            {
                slice = slice.split(",").map(function(index) {
                    index = +index;
                    if (0 > index) index += shape[i];
                    return index;
                });
            }
            else
            {
                slice = ([+slice]).map(function(index) {
                    if (0 > index) index += shape[i];
                    return index;
                });
            }
        }
        else if (is_num(slice, true))
        {
            slice = ([slice]).map(function(index) {
                if (0 > index) index += shape[i];
                return index;
            });
        }
        else if (is_array(slice, true))
        {
            //slice = slice.length > 2 ? {start:slice[0], end:slice[2], step:slice[1]} : {start:slice[0], end:slice[1], step:1};
            slice = slice.map(function(index) {
                index = +index;
                if (0 > index) index += shape[i];
                return index;
            });
        }
        else if ((null == slice.start) && (null == slice.end))
        {
            slice = {start:0, end:shape[i]-1, step:1};
        }
        if (is_array(slice, true))
        {
            slice = slice.filter(function(index) {
                return 0 <= index && index < shape[i];
            });
        }
        else
        {
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
        }
        return slice;
    });
}
function compute_slice_length(length, slice)
{
    if (is_array(slice, true))
    {
        return !length ? 0 : (slice.length);
    }
    else
    {
        if (!length || (0 > slice.step && (slice.start < 0 || slice.start < slice.end)) || (0 < slice.step && (slice.start >= length || slice.start > slice.end))) return 0;
        return stdMath.min(length, stdMath.ceil((stdMath.abs(slice.end-slice.start)+1)/stdMath.abs(slice.step)));
    }
}
function is_perm(p, n)
{
    if (p.length !== n) return false;
    var i, cnt = array(n, 0);
    for (i=0; i<n; ++i)
    {
        if ((0 > p[i]) || (p[i] >= n)) return false;
        ++cnt[p[i]];
    }
    return cnt.filter(function(cnt) {return 1 === cnt;}).length === n;
}
function invperm(p)
{
    var i, n = p.length, ip = new Array(n);
    for (i=0; i<n; ++i) ip[p[i]] = i;
    return ip;
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
    if (is_array(x, true))
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
