"use strict";

const TensorView = require('../src/TensorView.js');

const array = [1,2,3,4,5,6]; // single-dimensional data
const ndarray = [[1,2,3],[4,5,6]]; // multi-dimensional data

const s = TensorView(array, {shape:[2,3]}); // create a view with shape
const sT = s.transpose(); // get transposed view

const m = TensorView(ndarray, {ndarray:[2,3],shape:[3,2]}); // create view of ndarray with different shape
const m2 = m.reshape([2,3]).slice([null,[1,2,1]]); // reshape and get a slice

console.log(s.toNDArray());
console.log(s.toArray());
console.log(sT.toNDArray());
console.log(sT.toArray());
console.log(s.data() === sT.data()) // uses same data
console.log(m.toNDArray());
console.log(m.toArray());
console.log(m2.toNDArray());
console.log(m2.toArray());
console.log(m.data() === m2.data()) // uses same data

// iterator protocol
for (let [data_i, i] of s) console.log([data_i, i.slice()]); // i is multi-dimensional index in general

// same as
s.forEach((data_i, i) => console.log([data_i, i.slice()])); // i is multi-dimensional index in general
