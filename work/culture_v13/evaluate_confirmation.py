"""Frozen inference first, answer-key scoring only in a separate later command."""
import datetime
import hashlib
import json
from common import ROOT,PRIVATE,OLD,read,save,token_count,recovered_output,sha,usage_complete,usage_upper_bound
from run_confirmation import verify_lock,visible_cases
from feature_tools_v13 import FeatureTools
from inference_v13 import predict,qloo_review_evidence_available
from metrics_v13 import summarize_predictions


def optional(path):
    return read(path) if path.exists() else {}


def complete_usage(record):
    return usage_complete(record)


def upper_usage(record):
    return usage_upper_bound(record)


def branch_cost(names,initial,short,hierarchy,plan,mode):
    bt=token_count(initial);reported=bt;upper=upper_usage(initial)
    complete=complete_usage(initial)
    if any(n.startswith('hier_') for n in names):
        row=hierarchy[mode]
        if not plan:raise ValueError('Missing shared planner record cannot be charged as zero')
        reported+=plan.get('reported_tokens',token_count(plan))+row.get('reported_tokens',token_count(row))
        pieces=[upper,upper_usage(plan),upper_usage(row)]
        upper=sum(pieces) if all(p is not None for p in pieces) else None
        complete=complete and complete_usage(plan) and complete_usage(row)
    if any(n.startswith('judge_') for n in names):
        row=short[mode];reported+=token_count(row)
        # A bounded single attempt has an explicit charged upper bound. A
        # missing transport usage without such a bound remains unknown.
        ru=upper_usage(row)
        upper=upper+ru if upper is not None and ru is not None else None
        complete=complete and complete_usage(row)
    return {'tokens_reported':reported,'tokens_reported_complete':complete,
            'tokens_upper_bound':upper if complete or upper is not None else None}


def predict_matched_control(features,initial,config,short,hierarchy):
    """Use the separately D/E-calibrated control through production inference.

    Its feature names are already mapped to CF and agent reviews. Neither the
    Qloo weights/scales nor Qloo availability are inherited by this control.
    Failed control reviews retain the shared initial-opinion fallback policy.
    """
    matched=config['matched_no_qloo']
    local={'selected_agent':'matched_no_qloo','selected_qloo':'matched_no_qloo',
           'models':{'matched_no_qloo':matched}}
    result=predict(features,initial,local,short,hierarchical=hierarchy,qloo_available=True)
    return {'choice':result['choices']['agent'],
            'probability':result['probabilities']['agent'],
            'description':'Matched mechanism with Qloo removed or replaced by CF; independently recalibrated on D/E using the same nested ridge rule before confirmation.',
            'source_family':matched['source_family'],'reused_from':matched['reused_from'],
            'review_fallbacks':{'short':result['judge_fallbacks']['agent'],
                                'hierarchical':result['hierarchical_fallbacks']['agent']}}


def infer():
    verify_lock();cfg=read(ROOT/'agent_config.json');tool=FeatureTools()
    agent_names=cfg['models'][cfg['selected_agent']]['features']
    qloo_names=cfg['models'][cfg['selected_qloo']]['features']
    matched_names=cfg['matched_no_qloo']['features']
    required=set(agent_names+qloo_names+matched_names)
    for c in visible_cases():
        ident=c['id'];initial=read(PRIVATE/'predictions'/f'{ident}.json')
        short={m:optional(PRIVATE/'judges'/cfg['judge_version']/m/f'{ident}.json') for m in ['agent','qloo']}
        hv=cfg.get('hierarchical_version')
        hierarchy={m:optional(PRIVATE/hv/'reviews'/m/f'{ident}.json') if hv else {} for m in ['agent','qloo']}
        plan=optional(PRIVATE/hv/'plans'/f'{ident}.json') if hv else {}
        for mode,names in [('agent',agent_names),('qloo',qloo_names)]:
            if any(n.startswith('judge_') for n in names):assert short[mode],f'Missing short review {ident}/{mode}'
            if any(n.startswith('hier_') for n in names):assert hierarchy[mode],f'Missing hierarchical review {ident}/{mode}'
        if 'judge_agent' in matched_names:assert short['agent'],f'Missing matched no-Qloo short review {ident}'
        if 'hier_agent' in matched_names:assert hierarchy['agent'],f'Missing matched no-Qloo hierarchical review {ident}'
        evidence=optional(PRIVATE/'evidence'/f'{ident}.json')
        group=optional(PRIVATE/'grouped_qloo/cases'/f'{ident}.json')
        built=tool.build(c['likes']+c['dislikes'],c['candidates'],initial,evidence or None,
                         required_features=required,group_record=group or None)
        available=built['trace']['source_availability']['required_tools_valid']
        if 'hier_qloo' in qloo_names:
            available=available and qloo_review_evidence_available(hierarchy['qloo'],hierarchical=True)
        if 'judge_qloo' in qloo_names:
            available=available and qloo_review_evidence_available(short['qloo'])
        result=predict(built['values'],initial,cfg,short,hierarchical=hierarchy,qloo_available=available)
        cost={'glm':branch_cost([],initial,short,hierarchy,plan,'agent'),
              'agent':branch_cost(agent_names,initial,short,hierarchy,plan,'agent'),
              'agent_qloo':branch_cost(qloo_names,initial,short,hierarchy,plan,'qloo')}
        matched=predict_matched_control(built['values'],initial,cfg,short,hierarchy)
        matched['cost']=branch_cost(matched_names,initial,short,hierarchy,plan,'agent')
        # Secondary intervention only: unlike the main mechanism ablation above,
        # this retains Qloo feature scales and weights and can shift input scale.
        ablated=dict(built['values'])
        for n in qloo_names:
            if n.startswith('interaction_mixed_'):
                source=n.replace('interaction_mixed_','interaction_cf_')
                ablated[n]=built['values'][source]
            elif n.startswith('bernoulli_') and n.endswith('_qloo'):
                source=n[:-5]+'_cf'
                ablated[n]=built['values'][source]
            elif n in {'signed_history_log_ratio','mf_weighted_qloo'} or n.startswith(('qmeta_','qgraph_','qgroup_')):
                ablated[n]=0.
        ah=dict(hierarchy);ah['qloo']=hierarchy.get('agent',{})
        aj=dict(short);aj['qloo']=short.get('agent',{})
        # The no-Qloo intervention does not inherit a Qloo outage. Its own
        # matched review can still succeed; any review failure uses the same
        # initial-opinion fallback through predict().
        ab=predict(ablated,initial,cfg,aj,hierarchical=ah,qloo_available=True)
        record={'id':ident,'candidates':c['candidates'],'decision':result,
                'features':built['values'],'trace':built['trace'],'cost':cost,
                'recovery':{mode:{key:hierarchy[mode].get(key) for key in
                            ['repair_eligible','repair_attempted','repair_succeeded','repair_reason','first_review_error']}
                            for mode,names in [('agent',agent_names),('qloo',qloo_names)]
                            if any(n.startswith('hier_') for n in names)},
                'ablation':matched,
                'secondary_fixed_weight_ablation':{'choice':ab['choices']['agent_qloo'],
                            'probability':ab['probabilities']['agent_qloo'],
                            'cost':matched['cost'],
                            'description':'Secondary diagnostic only: fixed Qloo fusion weights/scales, matched no-Qloo review and CF substitution, affinity evidence removed; susceptible to distribution shift.'}}
        save(PRIVATE/'confirmation_decisions'/f'{ident}.json',record)
    print('Frozen predictions saved for all100; no labels loaded or scores computed.')


def score():
    verify_lock();cfg=read(ROOT/'agent_config.json');cs=read(PRIVATE/'confirmation_cases.json')
    rows=[];ablation=[];secondary_ablation=[];fallback=[];decisions=[]
    for c in cs:
        r=read(PRIVATE/'confirmation_decisions'/f"{c['id']}.json");decisions.append(r)
        target=0 if c['candidates'][0]==c['preferred'] else 1
        for mode in ['glm','agent','agent_qloo']:
            choice=r['decision']['choices'][mode]
            row={'case_id':c['id'],'method':mode,'difficulty':c['difficulty'],
                 'correct':choice==target,'valid':choice is not None,**r['cost'][mode]}
            rows.append(row)
            if mode=='agent_qloo':
                ablation.append({**row,'method':'with_qloo'})
                secondary_ablation.append({**row,'method':'with_qloo'})
        ablation.append({'case_id':c['id'],'method':'without_qloo','difficulty':c['difficulty'],
                         'correct':r['ablation']['choice']==target,'valid':r['ablation']['choice'] is not None,
                         **r['ablation']['cost']})
        secondary=r['secondary_fixed_weight_ablation']
        secondary_ablation.append({'case_id':c['id'],'method':'without_qloo','difficulty':c['difficulty'],
                         'correct':secondary['choice']==target,'valid':secondary['choice'] is not None,
                         **secondary['cost']})
        if r['decision']['qloo_fallback']:fallback.append(c['id'])
    expected=[{'case_id':c['id'],'difficulty':c['difficulty']} for c in cs]
    report=summarize_predictions(rows,expected_cases=expected,methods=['glm','agent','agent_qloo'],
         comparisons=[('glm','agent'),('agent','agent_qloo'),('glm','agent_qloo')])
    report['ablation']={'status':'complete','description':decisions[0]['ablation']['description'],
         'kind':'development_recalibrated_matched_no_qloo',
         'configuration':{k:v for k,v in cfg['matched_no_qloo'].items() if k!='model'},
         'metrics':summarize_predictions(ablation,baseline_method='without_qloo',expected_cases=expected,
           methods=['without_qloo','with_qloo'],comparisons=[('without_qloo','with_qloo')])}
    report['secondary_fixed_weight_ablation']={'status':'complete',
         'description':decisions[0]['secondary_fixed_weight_ablation']['description'],
         'kind':'fixed_weight_distribution_shift_diagnostic',
         'metrics':summarize_predictions(secondary_ablation,baseline_method='without_qloo',expected_cases=expected,
           methods=['without_qloo','with_qloo'],comparisons=[('without_qloo','with_qloo')])}
    report.update(status='complete',sample_size=len(cs),selected_version='culture_v13',evaluation_status='one frozen confirmation')
    report['metadata']={'status':'complete','title':'TasteTrail：文化偏好 Agent 的独立100例检验',
        'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model':'glm-5.3-flash',
        'dataset':'MovieLens1M：100名此前未使用者，严格时间边界，20/40/40互斥难度分组',
        'protocol':'先用219开发案例选定方法；固定配置与代码后，独立100例一次检验，所有失败保留在分母。',
        'config_sha256':sha(ROOT/'agent_config.json'),'data_lock_sha256':sha(PRIVATE/'confirmation_cases.json'),
        'source_sha256':hashlib.sha256(json.dumps(read(ROOT/'research/formal_lock.json')['source_sha'],sort_keys=True).encode()).hexdigest(),
        'selected_agent':cfg['selected_agent'],'selected_qloo':cfg['selected_qloo'],
        'hierarchical_version':cfg.get('hierarchical_version'),'recovery_policy':cfg.get('recovery_policy'),
        'method_descriptions':{
            'glm':'一次GLM回答；输入相同的5部喜欢、3部不喜欢及两个候选。',
            'agent':'开发集选定的最强无Qloo控制：'+cfg['selected_agent']+'。实际融合特征：'+', '.join(cfg['models'][cfg['selected_agent']]['features'])+'。',
            'agent_qloo':'独立按相同开发规则选出的Qloo方案：'+cfg['selected_qloo']+'。实际融合特征：'+', '.join(cfg['models'][cfg['selected_qloo']]['features'])+'。结构未必与主表Agent完全相同；另用相同机制移除Qloo，并在开发集重新校准，检验文化信息增量。'},
        'limitations':['这是历史电影二选一偏好代理任务，不代表城市行程的真人满意度。',
            '70%是项目目标，不是比赛官方达标线；100例不足以精确判断几个百分点的差异。',
            '开发阶段尝试过多个方案；开发最高分不能当独立效果。',
            '模型服务只提供别名，未提供可核对的权重版本指纹。',
            '严格时间排序只约束目标用户的可见历史与候选评分；背景3640名用户使用全期评分，Qloo采用当前资料快照，因此这是回顾性偏好检验，不是全局按当年可获取信息的历史部署回测。',
            '表中token按单条生产分支计量；实际研究还运行了多个候选和消融，研究总开销更高。确认收集遇到数值Qloo故障时仍可能保留匹配复核调用，费用比线上提前回退更保守。',
            '主要机制消融在开发集重新校准无Qloo控制，其结构由开发集入选Qloo家庭决定；仍属条件式比较，不能替代主表独立选择的强Agent对照。',
            '固定权重去Qloo仅保留为次要诊断；它没有重新训练融合器，输入分布可能改变，不能据此单独声称Qloo增益。'],
        'official_rubric':{'source_url':'https://qloo.devpost.com/',
                           'dimensions':['Implementation','Design','Potential impact','Quality of the idea']}}
    report['qloo_fallback_count']=len(fallback)
    report['recovery']={mode:{'attempted':sum(bool(r.get('recovery',{}).get(mode,{}).get('repair_attempted')) for r in decisions),
                              'succeeded':sum(bool(r.get('recovery',{}).get(mode,{}).get('repair_succeeded')) for r in decisions)}
                        for mode in ['agent','qloo']}
    if cfg.get('recovery_policy'):
        report['metadata']['limitations'].append('层级复核两支均执行预先登记的格式/截断恢复：仅客观无效时最多追加一次low复核，保留首次失败费用，总预算18×；合法答案不因正确性或低置信度重试。旧v2与统一恢复后的开发结果同时保留。')
    report['selection']=read(ROOT/'research/final_development_selection.json')
    save(ROOT/'research/confirmation_report.json',report)
    save(PRIVATE/'confirmation_scored_rows.json',rows)
    print(json.dumps({'methods':report['methods'],'comparisons':report['comparisons'],
                      'fallback_count':len(fallback)},ensure_ascii=False,indent=2))
