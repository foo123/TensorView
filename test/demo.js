"use strict";

const TensorView = require('../src/TensorView.js');

const array = [1,2,3,4,5,6,7,8,9,0]; // single-dimensional data
const ndarray = [[1,2,3,4,5],[6,7,8,9,0]]; // multi-dimensional data

// transposes
const s = TensorView(array, {shape:[2,5]}); // create a view with shape
const sT = s.transpose(); // get transposed view
console.log('transposes');
console.log(s.toNDArray());
console.log(s.toArray());
console.log(sT.toNDArray());
console.log(sT.toArray());
console.log(s.data === sT.data) // uses same data

// iterator protocol
for (let [data_i, i] of s) console.log([data_i, i.slice()]); // i is multi-dimensional index in general

// same as
s.forEach((data_i, i) => console.log([data_i, i.slice()])); // i is multi-dimensional index in general

// slices and nested slices
const s1 = TensorView(array, {shape:[2,5]});
const s2 = s1.slice(':','2:4');
const s3 = s2.slice('1', '-1:-1:0').squeeze();
console.log('slices');
console.log(s1.toNDArray());
console.log(s2.toNDArray());
console.log(s3.toNDArray());
console.log(s1.data === s2.data, s1.data === s3.data); // uses same data

// concatenations
const c1 = TensorView(array, {shape:[2,5]});
const c2 = c1.concat(c1, 0);
const c3 = c2.slice(':', '2:-1:1').squeeze(); // get a slice and squeeze
console.log('concatenations');
console.log(c1.toNDArray());
console.log(c2.toNDArray());
console.log(c3.toNDArray());

const m = TensorView(ndarray, {shape:[5,2]}); // create view of ndarray with different shape
const m2 = m.reshape([2,5]); // reshape
const m3 = m2.slice(':', '1:2'); // get a slice
console.log('reshape');
console.log(m.toNDArray());
console.log(m.toArray());
console.log(m2.toNDArray());
console.log(m2.toArray());
console.log(m3.toNDArray());
console.log(m3.toArray());
console.log(m.data === m2.data, m.data === m3.data) // uses same data
