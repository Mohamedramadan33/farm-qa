import { TEAM, TEAM_KEYS, type TeamMemberKey } from '../types/qa';

export const ALIASES: Record<string, TeamMemberKey> = {
  eslam: 'islam',
  'islam.ashraf': 'islam',
};

export const DEFAULT_PASSWORD = '000000';

export function displayName(key: string | undefined | null): string {
  if (!key) return '—';
  const k = key as TeamMemberKey;
  return TEAM[k]?.name ?? key;
}

export function resolveUser(ident: string): TeamMemberKey | null {
  const i = String(ident || '').trim().toLowerCase();
  if (!i) return null;
  if (i in TEAM) return i as TeamMemberKey;
  if (i in ALIASES) return ALIASES[i];
  for (const k of TEAM_KEYS) {
    const email = TEAM[k].email.toLowerCase();
    if (i === email || i === email.split('@')[0]) {
      return k;
    }
  }
  return null;
}

export interface UnvalidatedFarmItem {
  subId: string;
  farmCode: string;
  governorate: string;
}

export interface DivideWorkResult {
  workerFarms: Record<TeamMemberKey, Array<{ subId: string; farmCode: string; governorate: string }>>;
  workerGovs: Record<TeamMemberKey, string[]>;
  logs: string[];
}

/**
 * Port of work_divider.py divide_work_fairly with governorate block chunking (max_gov_ratio=1.4)
 */
export function divideWorkFairly(
  unvalidatedFarms: UnvalidatedFarmItem[],
  activeWorkers: TeamMemberKey[] = TEAM_KEYS,
  maxGovRatio = 1.4
): DivideWorkResult {
  const logs: string[] = [];
  logs.push('[FAST RUN] جاري جلب وتوزيع البيانات على أعضاء الفريق...');

  const workerFarms: Record<TeamMemberKey, Array<{ subId: string; farmCode: string; governorate: string }>> = {
    mohamed: [],
    mansy: [],
    haidy: [],
    islam: [],
  };
  const workerGovs: Record<TeamMemberKey, string[]> = {
    mohamed: [],
    mansy: [],
    haidy: [],
    islam: [],
  };
  const workerCounts: Record<TeamMemberKey, number> = {
    mohamed: 0,
    mansy: 0,
    haidy: 0,
    islam: 0,
  };

  if (!activeWorkers.length) {
    return { workerFarms, workerGovs, logs };
  }

  const govGroups = new Map<string, UnvalidatedFarmItem[]>();
  for (const item of unvalidatedFarms) {
    const gov = item.governorate && item.governorate !== '—' ? item.governorate : 'غير محدد';
    const arr = govGroups.get(gov) ?? [];
    arr.push(item);
    govGroups.set(gov, arr);
  }

  const totalFarms = unvalidatedFarms.length;
  const numWorkers = activeWorkers.length;
  const targetShare = numWorkers > 0 ? totalFarms / numWorkers : 0;

  interface AssignmentBlock {
    blockLabel: string;
    realGov: string;
    items: UnvalidatedFarmItem[];
  }

  const assignmentBlocks: AssignmentBlock[] = [];
  for (const [gov, items] of govGroups.entries()) {
    if (items.length > targetShare * maxGovRatio && numWorkers > 1) {
      const chunkSize = Math.max(1, Math.floor(targetShare));
      for (let i = 0; i < items.length; i += chunkSize) {
        const chunkItems = items.slice(i, i + chunkSize);
        const partNum = Math.floor(i / chunkSize) + 1;
        assignmentBlocks.push({
          blockLabel: `${gov} (P${partNum})`,
          realGov: gov,
          items: chunkItems,
        });
      }
    } else {
      assignmentBlocks.push({ blockLabel: gov, realGov: gov, items });
    }
  }

  assignmentBlocks.sort((a, b) => {
    if (b.items.length !== a.items.length) return b.items.length - a.items.length;
    return a.blockLabel.localeCompare(b.blockLabel);
  });

  for (const block of assignmentBlocks) {
    const sortedWorkers = [...activeWorkers].sort((w1, w2) => {
      if (workerCounts[w1] !== workerCounts[w2]) return workerCounts[w1] - workerCounts[w2];
      return TEAM[w1].email.localeCompare(TEAM[w2].email);
    });
    const targetWorker = sortedWorkers[0];

    for (const item of block.items) {
      workerFarms[targetWorker].push({
        subId: item.subId,
        farmCode: item.farmCode,
        governorate: block.realGov,
      });
    }
    if (!workerGovs[targetWorker].includes(block.realGov)) {
      workerGovs[targetWorker].push(block.realGov);
    }
    workerCounts[targetWorker] += block.items.length;
  }

  logs.push(`عدد الاستمارات المتبقية للتوزيع: ${totalFarms}`);
  for (const w of activeWorkers) {
    const farms = workerFarms[w];
    const govCounts: Record<string, number> = {};
    for (const f of farms) {
      govCounts[f.governorate] = (govCounts[f.governorate] || 0) + 1;
    }
    const details = Object.entries(govCounts)
      .map(([gov, cnt]) => `${gov}: ${cnt}`)
      .join(', ');
    logs.push(` -> ${TEAM[w].email.split('@')[0]} (${farms.length}) -> [${details || 'لا يوجد'}]`);
  }

  return { workerFarms, workerGovs, logs };
}
