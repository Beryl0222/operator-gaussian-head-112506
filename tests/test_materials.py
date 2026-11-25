"""检查合成资料的关键业务关系，避免错误输入被误作有效资料。"""
import unittest
import numpy as np
from inspect_data import load_materials

class MaterialIntegrity(unittest.TestCase):
    def test_domain_relationships(self):
        v,a=load_materials()
        n,k,d=a['mean'].shape
        self.assertEqual(a['logits'].shape,(n,k))
        self.assertEqual(a['raw_lower'].shape,(n,k,d,d))
        self.assertEqual(len(v['objects']['records']),n)
        origins={m['provenance'] for r in v['objects']['records'] for m in r['measurements']}
        self.assertEqual(origins,{'measured','predicted'})
        self.assertTrue(any(not r['measurements'] for r in v['objects']['records']))

if __name__ == '__main__':
    unittest.main()
