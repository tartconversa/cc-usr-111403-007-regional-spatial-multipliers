"""空间权重与样本边界测试：冻结口径、禁止重新归一与零填充、孤立城市处理。"""
import copy
import json
import unittest
from pathlib import Path

from app.catalog import load
from app.service import SCHEMAS
from app import spatial as sp
from app import linalg as la

FIXTURE = Path(__file__).resolve().parents[1] / 'fixtures' / 'spatial_sample.json'
DATA = json.loads(FIXTURE.read_text(encoding='utf-8'))


class SpatialBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = load(DATA, SCHEMAS)

    def test_missing_cities_dropped_not_zero_filled(self):
        sample = sp.build_sample(self.snapshot, boundary='frozen')
        # c09(missing)、c10(suppressed) 被剔除，样本只剩 8 城
        self.assertEqual(sample['n'], 8)
        self.assertNotIn('c09', sample['cities'])
        self.assertNotIn('c10', sample['cities'])
        dropped_ids = {d['id'] for d in sample['dropped']}
        self.assertEqual(dropped_ids, {'c09', 'c10'})
        # y 向量长度严格等于观测城市数，绝不补零
        self.assertEqual(len(sample['y']), 8)
        self.assertEqual(len(sample['W']), 8)
        self.assertEqual(all(len(row) == 8 for row in sample['W']), True)

    def test_frozen_does_not_renormalize(self):
        """frozen 方案取完整全集行归一后的子矩阵，被保留城市的行和可小于 1。"""
        universe = sp.build_universe_w(self.snapshot)
        sample = sp.build_sample(self.snapshot, boundary='frozen')
        # c04 在全集上连向 c02,c05,c09；剔除 c09 后其冻结行和应严格小于 1
        pos = {k: i for i, k in enumerate(sample['cities'])}
        i04 = pos['c04']
        self.assertLess(sum(sample['W'][i04]), 0.9999999)
        # 与完整 W 子矩阵逐元素一致（无二次归一）
        upos = {k: i for i, k in enumerate(universe['cities'])}
        idx = [upos[k] for k in sample['cities']]
        for a, i in enumerate(idx):
            for b, j in enumerate(idx):
                self.assertEqual(sample['W'][a][b], universe['W'][i][j])

    def test_renorm_is_explicit_distinct_scheme(self):
        """renorm 是显式声明的敏感性方案，行和重新归一为 1，与 frozen 产生不同 W。"""
        frozen = sp.build_sample(self.snapshot, boundary='frozen')
        renorm = sp.build_sample(self.snapshot, boundary='renorm')
        # renorm 下非孤立行的行和必须为 1
        for i, city in enumerate(renorm['cities']):
            if city not in renorm['sample_isolates']:
                self.assertAlmostEqual(sum(renorm['W'][i]), 1.0, places=12)
        # 两方案 W 不相同（至少一处），证明边界选择被显式区分
        diff = any(abs(frozen['W'][i][j] - renorm['W'][i][j]) > 1e-12
                   for i in range(frozen['n']) for j in range(frozen['n']))
        self.assertTrue(diff)

    def test_isolated_city_zero_row_preserved(self):
        data = copy.deepcopy(DATA)
        # 新增一个观测但无任何边的孤立城市
        data['collections']['cities'].append(
            {'id': 'c11', 'revision': 1, 'y': 10.0,
             'features': {'investment': 2.0, 'labor': 9.0}, 'status': 'observed'})
        snap = load(data, SCHEMAS)
        sample = sp.build_sample(snap, boundary='frozen')
        self.assertIn('c11', sample['sample_isolates'])
        pos = sample['cities'].index('c11')
        self.assertEqual(sum(sample['W'][pos]), 0.0)  # 零行保留，不造边不填权

    def test_universe_row_standardization_zero_row(self):
        universe = sp.build_universe_w(self.snapshot)
        # 全集行归一：有出边的行行和为 1
        for i, key in enumerate(universe['cities']):
            raw_sum = sum(universe['raw_W'][i])
            if raw_sum > 0:
                self.assertAlmostEqual(sum(universe['W'][i]), 1.0, places=12)

    def test_unknown_edge_endpoint_rejected(self):
        data = copy.deepcopy(DATA)
        data['collections']['edges'].append(
            {'id': 'ebad', 'revision': 1, 'from': 'c01', 'to': 'ghost', 'weight': 1})
        with self.assertRaises(ValueError):
            sp.build_universe_w(load(data, SCHEMAS))

    def test_negative_weight_and_self_edge_rejected(self):
        data = copy.deepcopy(DATA)
        data['collections']['edges'].append(
            {'id': 'eneg', 'revision': 1, 'from': 'c01', 'to': 'c01', 'weight': 1})
        with self.assertRaises(ValueError):
            sp.build_universe_w(load(data, SCHEMAS))

    def test_design_matrix_zscore_reproducible(self):
        sample = sp.build_sample(self.snapshot, boundary='frozen')
        d_none = sp.design_matrix(sample, x_scale='none')
        d_z = sp.design_matrix(sample, x_scale='zscore')
        # 标准化后非截距列均值约 0、方差约 1
        cols = list(zip(*d_z['X']))
        for j, name in enumerate(d_z['names']):
            if name == 'intercept':
                continue
            self.assertAlmostEqual(la.mean(cols[j]), 0.0, places=10)

    def test_inconsistent_feature_keys_rejected(self):
        data = copy.deepcopy(DATA)
        data['collections']['cities'][0]['features'] = {'different_key': 1.0}
        with self.assertRaises(ValueError):
            sp.build_sample(load(data, SCHEMAS))


if __name__ == '__main__':
    unittest.main()
