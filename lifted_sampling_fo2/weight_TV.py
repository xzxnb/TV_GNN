from __future__ import annotations

import time
from sampling_fo2.wfomc import standard_wfomc, faster_wfomc, Algo, wfomc
from sampling_fo2.problems import MLNProblem
from sampling_fo2.fol.sc2 import  to_sc2
from sampling_fo2.fol.syntax import  top, AUXILIARY_PRED_NAME, QuantifiedFormula, Universal, Equivalence, bot
from sampling_fo2.fol.utils import new_predicate
from sampling_fo2.utils.polynomial import coeff_dict, create_vars, expand
from sampling_fo2.utils import Rational
from sampling_fo2.context import WFOMCContext
from sampling_fo2.fol.syntax import Pred
from sampling_fo2.parser.mln_parser import parse as mln_parse
from sampling_fo2.problems import WFOMCSProblem, MLN_to_WFOMC, MLN_to_WFOMC1
import pandas as pd
import itertools


# Construct a sentence from the mlnproblem. If hard_rule is True, construct the negation of the hard constraint;
# otherwise, construct an equivalent sentence normally.
def mln_sentence(mln: MLNProblem, hard_rule: bool = True, pred_new: str = AUXILIARY_PRED_NAME):
    weightings: dict[Pred, tuple[Rational, Rational]] = dict()
    if hard_rule:
        sentence = bot
        for weighting, formula in zip(*mln.rules):
            if weighting != float('inf'):
                continue
            free_vars = formula.free_vars()
            for free_var in free_vars:
                formula = QuantifiedFormula(Universal(free_var), formula)
            sentence = sentence | (~formula)
    else:
        sentence = top
        for weighting, formula in zip(*mln.rules):
            free_vars = formula.free_vars()
            if weighting != float('inf'):
                aux_pred = new_predicate(len(free_vars), pred_new)
                formula = Equivalence(formula, aux_pred(*free_vars))
                weightings[aux_pred] = (Rational(1, 1), Rational(1, 1))
            for free_var in free_vars:
                formula = QuantifiedFormula(Universal(free_var), formula)
            sentence = sentence & formula
    return [sentence, weightings]

def sentence_WFOMCSProblem(sentence1, weightings1, sentence2, weightings2, domain, cardinality_constraint = None):
    sentence = sentence1 & sentence2
    sentence = to_sc2(sentence)
    weightings = {**weightings1, **weightings2}
    return WFOMCSProblem(sentence, domain, weightings, cardinality_constraint)

def count_distribution_(context: WFOMCContext, preds1: list[Pred], preds2: list[Pred],
                       algo: Algo = Algo.STANDARD) \
        -> dict[tuple[int, ...], Rational]:

    pred2sym = {}
    preds = preds1+preds2
    syms = create_vars('x0:{}'.format(len(preds)))
    for sym, pred in zip(syms, preds):
        pred2sym[pred] = sym
    def get_weight(pred):
        if pred not in pred2sym.keys():
            return context.get_weight(pred)
        return context.get_weight(pred)[0]*pred2sym[pred], context.get_weight(pred)[1]

    if algo == Algo.STANDARD:
        res = standard_wfomc(
            context.formula, context.domain, get_weight
        )
    elif algo == Algo.FASTERv2:
        res = faster_wfomc(
            context.formula, context.domain, get_weight, True
        )
    symbols = [pred2sym[pred] for pred in preds]
    count_dist = {}
    res1 = expand(res)
    if context.decode_result(res1) == 0:
        return {(0, 0): Rational(0, 1)}
    for degrees, coef in coeff_dict(res1, symbols):
        count_dist[degrees] = coef
    return count_dist

# Output the TVdistance of the two mln nodes and the corresponding attributes (weight or average number of edges) of the two mln nodes.
def MLN_TV(mln1: str,mln2: str, w1, w2) :
    if mln1.endswith('.mln'):
        with open(mln1, 'r') as f:
            input_content = f.read()
        mln_problem1 = mln_parse(input_content)

    for i in range(len(mln_problem1.rules[1])):
        if mln_problem1.rules[0][i] == float('inf'):
            continue
        mln_problem1.rules[0][i] = w1

    wfomcs_problem11 = MLN_to_WFOMC1(mln_problem1, '@F')
    context11 = WFOMCContext(wfomcs_problem11)

    if mln2.endswith('.mln'):
        with open(mln2, 'r') as f:
            input_content = f.read()
        mln_problem2 = mln_parse(input_content)

    for i in range(len(mln_problem2.rules[1])):
        if mln_problem2.rules[0][i] == float('inf'):
            continue
        mln_problem2.rules[0][i] = w2

    wfomcs_problem22 = MLN_to_WFOMC1(mln_problem2, '@S')
    context22 = WFOMCContext(wfomcs_problem22)

    Z1 = wfomc(context11, Algo.STANDARD)
    Z2 = wfomc(context22, Algo.STANDARD)

    weights1: dict[Pred, tuple[Rational, Rational]]
    weights1_hard: dict[Pred, tuple[Rational, Rational]]
    weights2: dict[Pred, tuple[Rational, Rational]]
    weights2_hard: dict[Pred, tuple[Rational, Rational]]

    domain = mln_problem1.domain
    [sentence1, weights1] = mln_sentence(mln_problem1, False, 'F')
    [sentence2, weights2] = mln_sentence(mln_problem2, False, 'S')


    wfomcs_problem1 = sentence_WFOMCSProblem(sentence1, weights1, sentence2, weights2, domain)

    print('wfomcs_problem1: ', wfomcs_problem1)

    context1 = WFOMCContext(wfomcs_problem1)
    count_dist1 = count_distribution_(context1, list(weights1.keys()), list(weights2.keys()))

    print('count_dist1: ', count_dist1)
    res = Rational(0, 1)

    for key in count_dist1:
        w = w1**key[0]/Z1 - w2**key[1]/Z2
        res = res + abs(w * count_dist1[key])
    res = 0.5*res

    return res


if __name__ == '__main__':
    mln1 = "models\\k_colored_graph_1.mln"
    mln2 = "models\\k_colored_graph_1.mln"

    weight1 = [0.1*(i+1) for i in range(20)]
    weight2 = [0.1*(i+1) for i in range(20)]
    combinations = list(itertools.product(weight1, weight2))
    result = []

    w1 = create_vars("w1")
    w2 = create_vars("w2")

    start_time = time.time()
    res = MLN_TV(mln1, mln2, w1, w2)
    end_time = time.time()


    execution_time = end_time - start_time
    print(f"k_colored_graph_skip run time: {execution_time:.6f} s")


    start_time = time.time()
    for w in combinations:
        result.append([w[0], w[1], res.subs({w1: w[0], w2: w[1]})])
    end_time = time.time()
    execution_time = end_time - start_time

    print("code run time: ", execution_time)

    df = pd.DataFrame(result, columns=["w1", "w2", "TV"])
    excel_filename = "data1/color1_domain10_TV.xlsx"
    df.to_excel(excel_filename, index=False)







