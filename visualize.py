import sys
import genotypes
from graphviz import Digraph



def plot(genotype, filename):
  g = Digraph(
      format='pdf',
      edge_attr=dict(fontsize='150', fontname="times"),
      node_attr=dict(style='filled', shape='rect', align='center', fontsize='150', height='0.5', width='0.5', fontname="times"),
      engine='dot')
  #g.body.extend(['rankdir=LR', 'nodesep=5.0', 'ranksep=0.4', 'penwidth=8', 'arrowsize=12'])
  g.body.extend(['rankdir=LR', 'nodesep=2.5', 'ranksep=0.4', 'penwidth=8', 'arrowsize=12'])
  #g.body.extend(['rankdir=LR'])

  g.node("c_{k-2}", fillcolor='darkseagreen2', height='3', width='6', nodesep='5', penwidth='8')
  g.node("c_{k-1}", fillcolor='darkseagreen2', height='3', width='6', nodesep='5', penwidth='8')
  assert len(genotype) % 2 == 0
  steps = len(genotype) // 2

  for i in range(steps):
    g.node(str(i), fillcolor='lightblue', height='3', width='3', penwidth='8')

  for i in range(steps):
    for k in [2*i, 2*i + 1]:
      op, j = genotype[k]

      if j == 0:
        u = "c_{k-2}"
      elif j == 1:
        u = "c_{k-1}"
      else:
        u = str(j-2)
      v = str(i)
      # import code
      # code.interact(local=locals())
      op_list = op.split('_')
      if 'circle' in op_list:
        g.edge(u, v, label=op, fillcolor="gray", penwidth='8', arrowsize='4', fontcolor='red')
      else:
        g.edge(u, v, label=op, fillcolor="gray", penwidth='8', arrowsize='4')

  g.node("c_{k}", fillcolor='palegoldenrod', height='3', width='6', penwidth='8')
  for i in range(steps):
    g.edge(str(i), "c_{k}", fillcolor="gray", labeldistance='0.05', penwidth='8', arrowsize='4')

  g.render(filename, view=False)  #True


if __name__ == '__main__':
  if len(sys.argv) != 2:
    print("usage:\n python {} ARCH_NAME".format(sys.argv[0]))
    sys.exit(1)

  genotype_name = sys.argv[1]
  try:
    genotype = eval('genotypes.{}'.format(genotype_name))
  except AttributeError:
    print("{} is not specified in genotypes.py".format(genotype_name)) 
    sys.exit(1)

  plot(genotype.normal, "normal")
  plot(genotype.reduce, "reduction")

