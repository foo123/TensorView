/**
*  TensorView
*  View array data as multidimensional tensors of various shapes efficiently
*  @VERSION 1.0.0
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

var proto = 'prototype', stdMath = Math;

function TensorView(data, o, _)
{
    if (!(this instanceof TensorView)) return new TensorView(data, o, _);

    var self = this,
        is_transposed = false, is_value = false,
        op = null, refs = null, stack = null, stack_axis = -1,
        aux_indices = null, nd_shape = null, same_shape = false,
        ndim = 0, shape = null, stride = null, size = null,
        slicing = null, default_slicing = true, length = 0, total = 0, i;

    function compute_index(indices, ndim, transposed, shape, stride, size, slicing)
    {
        // compute single index for row/column-major ordering scheme from multidimensional indices
        var index = 0, axis, i;
        if (slicing)
        {
            for (axis=0; axis<ndim; ++axis)
            {
                i = indices[axis];
                if (0 > i) i += size[axis];
                if (0 > i || i >= size[axis]) throw "TensorView:index ("+indices[axis]+") for dimension ("+axis+") is out of bounds (0,"+(size[axis]-1)+")!";
                index += stride[axis] * (slicing[axis].start + i * slicing[axis].step);
            }
        }
        else
        {
            for (axis=0; axis<ndim; ++axis)
            {
                i = indices[axis];
                if (0 > i) i += size[axis];
                if (0 > i || i >= size[axis]) throw "TensorView:index ("+indices[axis]+") for dimension ("+axis+") is out of bounds (0,"+(size[axis]-1)+")!";
                index += stride[axis] * i;
            }
        }
        return index;
    }
    function compute_indices(index, ndim, transposed, shape, size, slicing, indices)
    {
        // compute multidimensional indices for row/column-major ordering scheme from single index
        indices = indices || new Array(ndim);
        var axis, i;
        if (transposed)
        {
            if (slicing)
            {
                for (axis=0; axis<ndim; ++axis)
                {
                    i = index % shape[axis];
                    index = stdMath.floor(index / shape[axis]);
                    indices[axis] = stdMath.floor((i - slicing[axis].start) / slicing[axis].step);
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
            if (slicing)
            {
                for (axis=ndim-1; axis>=0; --axis)
                {
                    i = index % shape[axis];
                    index = stdMath.floor(index / shape[axis]);
                    indices[axis] = stdMath.floor((i - slicing[axis].start) / slicing[axis].step);
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
    function walk(a, i, v)
    {
        var ai = a, n = i.length-1, j;
        for (j=0; j<n; ++j) ai = ai[i[j]];
        if (2 < arguments.length) ai[i[n]] = v;
        return ai[i[n]];
    }
    function get(index, indices)
    {
        if (refs)
        {
            if (op)
            {
                return refs[1] ? op(refs[0].get(indices), refs[1].get(indices), refs[0], refs[1], indices) : op(refs[0].get(indices), refs[0], indices);
            }
            else
            {
                return refs[0].get(refs[0].indices(index));
            }
        }
        else if (stack)
        {
            for (var i=0; i<ndim; ++i)
            {
                aux_indices[i] = slicing[i].start + indices[i] * slicing[i].step;
            }
            for (var i=0,sl=stack.length,t,tl; i<sl; ++i)
            {
                t = stack[i]; tl = t.size(stack_axis);
                if (0 <= aux_indices[stack_axis] && aux_indices[stack_axis] < tl) return t.get(aux_indices);
                aux_indices[stack_axis] -= tl;
            }
        }
        else if (nd_shape)
        {
            return walk(data, default_slicing && same_shape ? indices : compute_indices(index, nd_shape.length, false, nd_shape, nd_shape, null, aux_indices));
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
            if (op)
            {
                // nothing
            }
            else
            {
                refs[0].set(refs[0].indices(index), value);
            }
        }
        else if (stack)
        {
            for (var i=0; i<ndim; ++i)
            {
                aux_indices[i] = slicing[i].start + indices[i] * slicing[i].step;
            }
            for (var i=0,sl=stack.length,t,tl; i<sl; ++i)
            {
                t = stack[i]; tl = t.size(stack_axis);
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
            walk(data, default_slicing && same_shape ? indices : compute_indices(index, nd_shape.length, false, nd_shape, nd_shape, null, aux_indices), value);
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
        shape = o.shape || refs[0].size();
        ndim = shape.length;
    }
    else if (_ && _._refs && (_._refs[0] instanceof TensorView))
    {
        is_transposed = false;
        op = _._op || null;
        refs = _._refs;
        total = refs[0].length();
        shape = (op ? refs[0].size() : o.shape) || refs[0].size();
        ndim = shape.length;
        if (refs[1])
        {
            data = null;
        }
        if (op)
        {
            if (o.slice) o.slice = null;
        }
    }
    else if (_ && _._stack && (2 <= _._stack.length) && (_._stack[0] instanceof TensorView) && (_._stack[1] instanceof TensorView))
    {
        data = null;
        stack_axis = _._stack_axis || 0;
        stack = _._stack;
        shape = stack[0].size();
        shape[stack_axis] = sum(stack.map(function(t) {return t.size(stack_axis);}));
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
            same_shape = (nd_shape.length === shape.length) && (nd_shape.length === shape.filter(function(shapei,i) {return shapei === nd_shape[i];}).length);
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

    slicing = o.slice;
    if (!slicing || !slicing.length)
    {
        slicing = new Array(ndim);
        for (i=0; i<ndim; ++i) slicing[i] = new Slice(0, shape[i]-1, 1);
    }
    else
    {
        while (slicing.length < ndim) slicing.push(null);
        if (slicing.length > ndim) slicing.length = ndim;
        for (i=0; i<ndim; ++i)
        {
            slicing[i] = slicing[i] || [0,shape[i]-1,1];
            slicing[i] = Slice._indices(shape[i], slicing[i][0], slicing[i][1], slicing[i][2]);
            if (slicing[i].start !== 0 || slicing[i].stop+1 !== shape[i] || slicing[i].step !== 1) default_slicing = false;
        }
    }

    size = new Array(ndim);
    for (i=0; i<ndim; ++i) size[i] = slicing[i].count(shape[i]);
    length = ndim ? product(size) : 0;
    o = _ = null;

    self.dispose = function() {
        op = null;
        refs = null;
        stack = null;
        aux_indices = null;
        nd_shape = null;
        data = null;
        shape = null;
        stride = null;
        slicing = null;
        size = null;
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
    self.slicing = function(axis) {
        return arguments.length ? slicing[axis].toObj() : slicing.map(function(s) {return s.toObj();});
    };
    self.size = function(axis) {
        return arguments.length ? size[axis] : size.slice();
    };
    self.length = function() {
        return length;
    };
    self.index = function(/*indices*/) {
        return compute_index(Array.isArray(arguments[0]) ? arguments[0] : arguments, ndim, is_transposed, shape, stride, size, slicing);
    };
    self.indices = function(index) {
        if (0 > index || index >= total) throw "TensorView::indices:index ("+index+") is out of bounds (0,"+(total-1)+")!";
        return compute_indices(index, ndim, is_transposed, shape, size, slicing);
    };
    self.get = function(/*indices*/) {
        var indices = Array.isArray(arguments[0]) ? arguments[0] : arguments, index = 0;
        if (indices.length < ndim) throw "TensorView::get:indices do not match shape dimension!";
        if (!op && !stack) index = compute_index(indices, ndim, is_transposed, shape, stride, size, slicing);
        if (0 > index || index >= total) throw "TensorView::get:index ("+index+") is out of bounds (0,"+(total-1)+")!";
        return get(index, indices);
    };
    self.set = function(/*indices, value*/) {
        var indices = arguments, count = arguments.length-1, index = 0;
        if (Array.isArray(arguments[0])) {indices = arguments[0]; count = indices.length;}
        if (count < ndim) throw "TensorView::set:indices do not match shape dimension!";
        if (!op && !stack) index = compute_index(indices, ndim, is_transposed, shape, stride, size, slicing);
        if (0 > index || index >= total) throw "TensorView::set:index ("+index+") is out of bounds (0,"+(total-1)+")!";
        set(index, indices, arguments[arguments.length-1]);
        return self;
    };
    self.iterator = function() {
        var i = 0 < length ? ndim - 1 : -1,
            indices = null, ind = null, striding = null, index = 0,
            value = [null, null], ret = {value: null};
        return {next:function next() {
            if (0 > i)
            {
                indices = ind = striding = ret = value = null;
                return {done: true};
            }
            else
            {
                if (!indices)
                {
                    indices = (new Array(ndim)).fill(0);
                    ind = indices.slice();
                    striding = slicing.map(function(si,i) {return {start:si.start*stride[i], step:si.step*stride[i]};});
                    index = compute_index(indices, ndim, is_transposed, shape, stride, size, slicing);
                    value[0] = get(index, indices);
                    value[1] = ind;
                    ret.value = value;
                }
                else
                {
                    while (i >= 0 && indices[i]+1 >= size[i])
                    {
                        index -= striding[i].start + indices[i] * striding[i].step;
                        --i;
                    }
                    if (0 <= i)
                    {
                        ++indices[i];
                        ind[i] = indices[i];
                        index += striding[i].step;
                        while (i+1 < ndim)
                        {
                            ++i;
                            indices[i] = 0;
                            ind[i] = 0;
                            index += striding[i].start;
                        }
                        value[0] = get(index, indices);
                        value[1] = ind;
                        ret.value = value;
                    }
                    else
                    {
                        indices = ind = striding = ret = value = null;
                        return {done: true};
                    }
                }
                return ret;
            }
        }};
    };
    self.forEach = function(f) {
        if (0 < length && is_function(f))
        {
            var iter = self.iterator(), next, ret = null;
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
        return new TensorView(
        data,
        {
            ndarray: nd_shape,
            shape: shape.slice().reverse(),
            slice: slicing.map(function(s) {return s.toArr();}).reverse()
        },
        {
            _transposed: !is_transposed,
            _refs: refs ? refs.map(function(t) {return t.transpose();}).reverse() : null,
            _op: op,
            _stack: stack ? stack.map(function(t) {return t.transpose();}) : null,
            _stack_axis: stack ? ndim-1-stack_axis : -1
        }
        );
    };
    self.slice = function(slices) {
        return new TensorView(
        data,
        {
            ndarray: nd_shape,
            shape: op && refs ? null : shape,
            slice: op && refs ? null : slicing.map(function(slicei,i) {
                var si = slices[i] || [null,null,null];
                return slicei.subslice(Slice(si[0], si[1], si[2]).indices(shape[i])).toArr();
            })
        },
        {
            _transposed: is_transposed,
            _refs: op && refs ? refs.map(function(t) {return t.slice(slices);}) : refs,
            _op: op,
            _stack: refs ? null : stack,
            _stack_axis: stack_axis
        }
        );
    };
    self.reshape = function(shape) {
        return default_slicing && !stack && !refs ? new TensorView(data, {ndarray: nd_shape, shape: shape}) : new TensorView(self, {shape: shape});
    };
    self.concat = function(others, axis) {
        axis = axis || 0;
        if (others instanceof TensorView) others = [others];
        for (var i=0,n=others.length; i<n; ++i)
        {
            var matchSize = size.filter(function(sizej, j) {return j === axis || sizej === others[i].size(j);});
            if (matchSize.length !== size.length) throw "TensorView::concat:["+size.map(function(size,i){return axis===i?':':size;}).join(',')+"] and ["+others[i].size().map(function(size,i){return axis===i?':':size;}).join(',')+"] sizes do not match!";
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
    self.op = function(op, other) {
        /*if (other)
        {
            var matchSize = size.filter(function(sizej, j) {return sizej === other.size(j);});
            if (matchSize.length !== size.length) throw "TensorView::op:["+size.join(',')+"] and ["+other.size().join(',')+"] sizes do not match!";
        }*/
        return new TensorView(
        data,
        null,
        {
            _refs: other instanceof TensorView ? [self, other] : [self],
            _op: op
        }
        );
    };
    self.toArray = function(ArrayClass) {
        var array = new (ArrayClass || Array)(length), index = 0;
        self.forEach(function(di/*,i*/) {
            // put in row-major order
            array[index++] = di;
        });
        return array;
    };
    self.toNDArray = function() {
        var ndarray = ndim ? new Array(size[0]) : [];
        self.forEach(function(di, i) {
            // put in row-major order
            for (var a=ndarray,n=ndim-1,j=0,ij; j<n; ++j)
            {
                ij = i[j];
                if (null == a[ij]) a[ij] = new Array(size[j+1]);
                a = a[ij];
            }
            a[i[n]] = di;
        });
        return ndarray;
    };
    self.toString = function(maxsize) {
        var str = '', max = -Infinity, rows = null,
            rem = '', maxsize2, oversize, inlimits;
        if (null != maxsize)
        {
            maxsize2 = maxsize >>> 1;
            if (2 <= ndim)
            {
                oversize = size.map(function(size) {return maxsize < size;});
                inlimits = function(i, j) {
                    return !oversize[j] || (j <= ndim-3 && (i < maxsize2 || i > size[j]-1-maxsize2)) || (j > ndim-3 && (i <= maxsize2 || i >= size[j]-1-maxsize2));
                };
                rows = (new Array(oversize[size.length-2] ? maxsize+1 : size[size.length-2])).fill(null).map(function(_) {return new Array(oversize[size.length-1] ? maxsize+1 : size[size.length-1]);});
                self.forEach(function(di, i) {if (i.length === i.filter(inlimits).length) max = stdMath.max(String(di).length, max);});
                self.forEach(function(di, i) {
                    // print in 2d slices
                    if (i.length === i.filter(inlimits).length)
                    {
                        var i1 = i[ndim-2], i2 = i[ndim-1];
                        if (oversize[ndim-2] && i1 > maxsize2) i1 = maxsize - (size[ndim-2]-1-i1);
                        if (oversize[ndim-1] && i2 > maxsize2) i2 = maxsize - (size[ndim-1]-1-i2);
                        rows[i1][i2] = pad(String(di), max, ' ', false);
                        if (size[ndim-1] === 1+i[ndim-1] && size[ndim-2] === 1+i[ndim-2])
                        {
                            str += rem + rows.map(function(row, j) {
                                if (oversize[ndim-2] && maxsize2 === j) row = new Array(row.length).fill(pad(':', max, ' ', false));
                                if (oversize[ndim-1]) row[maxsize2] = pad('..', max, ' ', false);
                                return row.join(' ');
                            }).join("\n");
                            var j = ndim-3, o = -1;
                            while (0 <= j ) {o = (-1 === o) && oversize[j] && (i[j]+1 === maxsize2) ? j : o; --j;}
                            rem = -1 === o ? "\n-\n" : "\n-\n"+pad(':', max, ' ', false)+' '+(new Array((oversize[ndim-1] ? maxsize+1 : size[ndim-1])-2)).fill(pad('..', max, ' ', false)).join(' ')+' '+pad(':', max, ' ', false)+"\n-\n";
                        }
                    }
                });
            }
            else
            {
                // print in 1d slices
                str = maxsize >= size[0] ? self.toString() : (self.slice([{start:0,stop:maxsize2+1}]).toString() + ' .. ' + self.slice([{start:size[0]-1-maxsize2,stop:size[0]-1}]).toString());
            }
        }
        else
        {
            if (2 <= ndim)
            {
                rows = (new Array(size[size.length-2])).fill(null).map(function(_) {return new Array(size[size.length-1]);});
                self.forEach(function(di) {max = stdMath.max(String(di).length, max);});
                self.forEach(function(di, i) {
                    // print in 2d slices
                    rows[i[ndim-2]][i[ndim-1]] = pad(String(di), max, ' ', false);
                    if (size[ndim-1] === 1+i[ndim-1] && size[ndim-2] === 1+i[ndim-2])
                    {
                        str += rem + rows.map(function(row) {return row.join(' ');}).join("\n");
                        rem = "\n-\n";
                    }
                });
            }
            else
            {
                self.forEach(function(di) {
                    // print in 1d slices
                    str += (str.length ? " " : "") + String(di);
                });
            }
        }
        return str;
    };
}
TensorView.VERSION = '1.0.0';
TensorView[proto] = {
    constructor: TensorView,
    dispose: null,
    data: null,
    dimension: null,
    shape: null,
    slicing: null,
    size: null,
    length: null,
    index: null,
    indices: null,
    get: null,
    set: null,
    iterator: null,
    forEach: null,
    slice: null,
    reshape: null,
    concat: null,
    transpose: null,
    op: null,
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

function Slice(start, stop, step)
{
    var self = this;
    if (!(self instanceof Slice)) return new Slice(start, stop, step);
    self.start = null == start ? null : start;
    self.stop = null == stop ? null : stop;
    self.step = null == step ? null : step;
}
Slice[proto] = {
    constructor: Slice,
    start: null,
    stop: null,
    step: null,
    clone: function() {
        return new Slice(this.start, this.stop, this.step);
    },
    indices: function(length) {
        return Slice._indices(length, this.start, this.stop, this.step);
    },
    count: function(length) {
        return Slice._count(length, this.start, this.stop, this.step);
    },
    subslice: function(subslice) {
        return Slice._subslice(this, subslice);
    },
    toObj: function() {
        return {start:this.start, stop:this.stop, step:this.step};
    },
    toArr: function() {
        return [this.start, this.stop, this.step];
    }
};
Slice._indices = function(length, start, stop, step) {
    step = step || 1;
    if (null == start) start = 0;
    else start = (0 > start ? length : 0)+start;
    if (null == stop) stop = length-1;
    else stop = (0 > stop ? length : 0)+stop;
    stop = start + step*stdMath.floor(stdMath.abs(stop-start)/stdMath.abs(step));
    return new Slice(start, stop, step);
};
Slice._count = function(length, start, stop, step) {
    if (!length || (0 > step && (start < 0 || start < stop)) || (0 < step && (start >= length || start > stop))) return 0;
    return stdMath.min(length, stdMath.ceil((stdMath.abs(stop-start)+1)/stdMath.abs(step)));
};
Slice._subslice = function(slice, subslice) {
    //i0 : 0 -> size0-1, index0 = a0 + i0*s0
    //i : 0 -> size-1, index = a + i*s
    //a -> a0+s0*a, b same
    var step = slice.step || 1;
    return subslice ? new Slice(slice.start+(subslice.start||0)*step, slice.start+(subslice.stop)*step, (subslice.step||1)*step) : (slice instanceof Slice ? slice : new Slice(slice.start, slice.stop, step));
};
TensorView.slice = Slice;

// utils
function is_function(x)
{
    return "function" === typeof x;
}
function is_array(x)
{
    if (Array.isArray(x)) return true;
    else if (("undefined" !== typeof Float32Array) && (x instanceof Object.getPrototypeOf(Float32Array))) return true;
    return false;
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

// export it
return TensorView;
});
