"use strict";

const TensorView = require('../src/TensorView.js');
const echo = console.log;

const array = [1,2,3,4,5,6,7,8,9,0]; // single-dimensional data
const ndarray = [[1,2,3,4,5],[6,7,8,9,0]]; // multi-dimensional data

// transposes
const s = TensorView(array, {shape:[2,5]}); // create a view with shape
const sT = s.transpose(); // get transposed view
echo('transposes');
echo(s.toNDArray());
echo(s.toArray());
echo(sT.toNDArray());
echo(sT.toArray());
echo(s.data === sT.data) // uses same data

echo('---');

// iterator protocol
for (let [item, index] of s) echo([item, index.slice()]); // index is array of multidimensional indices
echo();
// same as
s.forEach((item, index) => echo([item, index.slice()])); // index is array of multidimensional indices
echo();
for (let [item, index] of sT) echo([item, index.slice()]); // index is array of multidimensional indices
echo();
// same as
sT.forEach((item, index) => echo([item, index.slice()])); // index is array of multidimensional indices

// slices, nested slices and selections
const s1 = TensorView(array, {shape:[2,5]});
const s2 = s1.slice(':','2:4');
const s3 = s2.slice('1', '-1:-1:0').squeeze();
const s4 = s2.slice('1', '-1,0').squeeze();
const s5 = s1.select([[0,0,1,0,0], [0,1,0,1,0]]);

echo('---');

echo('slices');
echo(s1.toNDArray());
echo(s2.toNDArray());
echo(s3.toNDArray());
echo(s4.toNDArray());
echo(s1.data === s2.data, s1.data === s3.data, s1.data === s4.data); // uses same data

echo('---');

echo('selections');
echo(s1.toArray());
echo(s5.toArray());
echo(s1.data === s5.data); // uses same data

s2.set([1,1],34);
echo('---');

echo('set');
echo(s1.toNDArray());
echo(s2.toNDArray());
echo(s3.toNDArray());
echo(s4.toNDArray());
echo(s1.data === s2.data, s1.data === s3.data, s1.data === s4.data); // uses same data

s2.setFrom([11,12,13,14,15,16])
s5.setFrom(345);
echo('---');

echo('block set');
echo(s1.toNDArray());
echo(s2.toNDArray());
echo(s3.toNDArray());
echo(s4.toNDArray());
echo(s5.select(false).toNDArray());
echo(s1.data === s2.data, s1.data === s3.data, s1.data === s4.data, s1.data === s5.data); // uses same data

// concatenations
const c1 = TensorView(array, {shape:[2,5]});
const c2 = c1.concat(c1, 0);
const c3 = c2.slice(':', '2:-1:1').squeeze(); // get a slice and squeeze
const A = TensorView([[1,2,3], [4,5,6], [7,8,9]]);
const B = TensorView([[0,5,4], [2,7,6], [9,2,1]]);
const C = A.concat(B, "newaxis");

echo('---');

echo('concatenations');
echo(c1.toNDArray());
echo(c2.toNDArray());
echo(c3.toNDArray());
echo('---');
echo(A.toString());
echo(B.toString());
echo(C.toString());

const P1 = C.permute([1,0,2]); // interchange rows/columns
echo('---');

echo('permutations');
echo(C.toString());
echo(P1.toString());

const m = TensorView(ndarray, {shape:[5,2]}); // create view of ndarray with different shape
const m2 = m.reshape([2,5]); // reshape
const m3 = m2.slice(':', '1:2'); // get a slice

echo('---');

echo('reshape');
echo(m.toNDArray());
echo(m.toArray());
echo(m2.toNDArray());
echo(m2.toArray());
echo(m3.toNDArray());
echo(m3.toArray());
echo(m.data === m2.data, m.data === m3.data) // uses same data

const a = TensorView([1,2,3,4,5,6], {shape:[2,3]});
const c = a.transpose();
const b = a.reorder("column-major", "row-major");
echo('---');

echo('reorder');
echo(a.toNDArray());
echo(b.toNDArray());
echo(c.toNDArray());
echo(a.toArray());
echo(b.toArray());
echo(c.toArray());
echo('a', {i:[0,1], idx:a.index([0,1]), v:a.get([0,1])});
echo('b', {i:[0,1], idx:b.index([0,1]), v:b.get([0,1])});
echo('c', {i:[0,1], idx:c.index([0,1]), v:c.get([0,1])});

const A1 = TensorView([["a11", "a12", "a13", "a14", "a15"], ["a21", "a22", "a23", "a24", "a25"], ["a31", "a32", "a33", "a34", "a35"]]);
const A2 = TensorView([["b11", "b12", "b13", "b14", "b15"], ["b21", "b22", "b23", "b24", "b25"], ["b31", "b32", "b33", "b34", "b35"]]);
const AA = A1.concat(A2, "newaxis");
const AA2 = AA.map(function(x) {return x+"_2";});
const AAA = AA.concat(AA2, "newaxis")
const AAO = AA.reorder("row-major", "column-major");
const AAT = AA.transpose();
const AAAO = AAA.reorder("row-major", "column-major");
const AAAT = AAA.transpose();
echo('---');

echo('concat/permute/reshape/reorder');
echo(AA.toString());
echo('---');
echo(AAO.toString());
echo('---');
echo(AAT.toString());
echo('---');
echo('AA ', {i:[0,1,1], idx:AA .index([0,1,1]), v:AA .get([0,1,1])});
echo('AAO', {i:[0,1,1], idx:AAO.index([0,1,1]), v:AAO.get([0,1,1])});
echo('AAT', {i:[0,1,1], idx:AAT.index([0,1,1]), v:AAT.get([0,1,1])});
echo('---');
echo(AA.reshape([6,5]).toString());
echo('---');
echo(AA.permute(1,0,2).reshape([6,5]).toString());
echo('---');
echo(AA.permute(2,0,1).reshape([6,5]).toString());
echo('---');
echo(AA.permute(2,1,0).reshape([6,5]).reorder([0,1],[1,0]).toString());
echo('---');
echo(AAA.reshape([12,5]).toString());
echo('---');
echo(AAA.permute(1,0,2,3).reshape([12,5]).toString());
echo('---');
echo(AAA.permute(3,2,0,1).reshape([12,5]).toString());
echo('---');
echo(AAA.permute(2,3,1,0).reshape([12,5]).reorder([0,1],[1,0]).toString());

const a1 = [[11, 12, 13, 14],[21, 22, 23, 24],[31, 32, 33, 34]];
const b1 = [[110, 120, 130, 140],[210, 220, 230, 240],[310, 320, 330, 340]];

const t1 = TensorView(a1).concat(TensorView(b1), "newaxis");
echo('---');
echo(t1.toString());
echo('---');
echo(t1.toArray().join("\n"));
echo('---');
echo(t1.permute(2,1,0).toArray().join("\n"));

echo('---');
const A11 = TensorView([["a11", "a12", "a13", "a14", "a15"], ["a21", "a22", "a23", "a24", "a25"], ["a31", "a32", "a33", "a34", "a35"]]);
const A21 = TensorView([["b11", "b12", "b13", "b14", "b15"], ["b21", "b22", "b23", "b24", "b25"], ["b31", "b32", "b33", "b34", "b35"]]);
const AA11 = A11.concat(A21, "newaxis");
const AA22 = TensorView(AA11.permute(2,1,0).toArray(), {shape:[5,6]}).permute(1,0);
echo(AA22.toString());
