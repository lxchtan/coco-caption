import argparse
from pycocoevalcap.eval import COCOEvalCap

class text(object):
  def __init__(self, filename):
    self.imgToAnns = {}

    with open(filename) as f:
      lines = f.readlines()
      for _id, line in enumerate(lines):
        self.imgToAnns[_id] = [{"caption": line}]

  def getImgIds(self):
    return self.imgToAnns.keys()

def cal_scores(ref, pred):
  ref_text = text(ref)
  pred_text = text(pred)

  evaluator = COCOEvalCap(ref_text, pred_text)

  evaluator.evaluate()

  for metric, score in evaluator.eval.items():
    print '%s: %.3f'%(metric, score*100)
  return evaluator

def run(pred, ref, result_write_to=None):
  print(pred)
  evaluator = cal_scores(ref, pred)
  if result_write_to is None:
    result_write_to = pred[:-4] + '_score.txt'
    with open(result_write_to, 'w') as f:
      for metric, score in evaluator.eval.items():
        f.write('%s: %.3f\n'%(metric, score*100))
  else:
    with open(result_write_to, 'a') as f:
      f.write(pred + '\n')
      for metric, score in evaluator.eval.items():
        f.write('%s: %.3f\n'%(metric, score*100))
      f.write('\n')     

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--score_file', default=None, type=str, help='Output score file.')
  parser.add_argument('--ref_file', required=True, type=str, help='Golden labels.')
  parser.add_argument('--generate_file', required=True, type=str, help='Generation results.')

  args = parser.parse_args()

  run(args.generate_file, args.ref_file, result_write_to=args.score_file)