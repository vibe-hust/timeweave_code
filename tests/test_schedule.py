import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from protocol import event_plan,layout

class ScheduleTests(unittest.TestCase):
    def test_original_intervals_are_not_clipped(self):
        row={'prompt':[[0,1,22],[30,28,24],['scene','long action','other'],[0,1,2]]}
        blocks=event_plan(row)
        self.assertEqual(blocks[0]['prompt'][0],[0,1])
        self.assertEqual(blocks[0]['prompt'][1],[30,28])
        self.assertIn(2,blocks[-3]['indices'])
    def test_globals_survive_every_block(self):
        row={'prompt':[[0,0],[1,60],['scene','action'],[0,1]]}
        self.assertTrue(all(0 in b['indices'] for b in event_plan(row,total=241)))
    def test_blocks_cover_exactly_requested_latents(self):
        for total in [41,121,241]:
            self.assertEqual([i for a,b in layout(total) for i in range(a,b)],list(range(total)))
    def test_missing_actions_fail(self):
        with self.assertRaises(ValueError):event_plan({'prompt':[[0],[30],['scene'],[0]]})
if __name__=='__main__':unittest.main()
