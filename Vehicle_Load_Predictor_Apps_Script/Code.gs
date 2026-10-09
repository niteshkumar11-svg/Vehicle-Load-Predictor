const SPREADSHEET_ID = '1SbLc5pt0YPDBEQVOaOfyd-AJfvhTthQ5zUAcGgFU7Tc';
const SHIPMENTS_PER_BAG = 30;
const PRESENCE_SHEET_NAME = 'Dashboard Active Users';
const PRESENCE_TTL_MS = 5 * 60 * 1000;

const VEHICLE_CFT = {
  '6.5 Ft': 6.5 * 4.65 * 4.75,
  '8 Ft': 8 * 5.0 * 5.25,
  '10 Ft': 10 * 5.75 * 5.75,
  '14 Ft': 14 * 6.0 * 6.25,
  '17 Ft': 17 * 6.5 * 6.75,
  '20 Ft': 20 * 7.5 * 7.5,
  '22 Ft': 22 * 7.5 * 7.5,
  '24 Ft': 24 * 7.75 * 7.75,
  '32 Ft': 32 * 8.0 * 9.0,
};

const CFT_PER_BAG = VEHICLE_CFT['32 Ft'] / (17000 / SHIPMENTS_PER_BAG);
const CFT_PER_SEMI = VEHICLE_CFT['32 Ft'] / 1800;
const CFT_PER_TOTE = VEHICLE_CFT['32 Ft'] / 650;

function doGet() {
  return HtmlService.createHtmlOutputFromFile('Index')
    .setTitle('Vehicle Load Predictor | Hajipur MH')
    .setXFrameOptionsMode(HtmlService.XFrameOptionsMode.ALLOWALL);
}

function registerActiveUser() {
  const ss = SpreadsheetApp.openById(SPREADSHEET_ID);
  const sheet = ss.getSheetByName(PRESENCE_SHEET_NAME) || ss.insertSheet(PRESENCE_SHEET_NAME);
  const email = activeUserEmail_();
  const now = new Date();
  const lock = LockService.getScriptLock();
  lock.waitLock(5000);
  try {
    if (sheet.getLastRow() === 0) sheet.appendRow(['User', 'Last Seen']);
    const lastRow = sheet.getLastRow();
    const values = lastRow > 1 ? sheet.getRange(2, 1, lastRow - 1, 2).getValues() : [];
    const existingIndex = values.findIndex(row => String(row[0] || '').toLowerCase() === email.toLowerCase());
    if (existingIndex >= 0) {
      sheet.getRange(existingIndex + 2, 2).setValue(now);
    } else {
      sheet.appendRow([email, now]);
    }
    return activeUsersFromSheet_(sheet, now);
  } finally {
    lock.releaseLock();
  }
}

function activeUserEmail_() {
  try {
    return Session.getActiveUser().getEmail() || 'Guest';
  } catch (error) {
    return 'Guest';
  }
}

function activeUsersFromSheet_(sheet, now) {
  const lastRow = sheet.getLastRow();
  if (lastRow <= 1) return [];
  const cutoff = now.getTime() - PRESENCE_TTL_MS;
  return sheet.getRange(2, 1, lastRow - 1, 2).getValues()
    .map(row => ({email: String(row[0] || 'Guest'), lastSeen: row[1]}))
    .filter(user => user.lastSeen instanceof Date && user.lastSeen.getTime() >= cutoff)
    .sort((left, right) => right.lastSeen.getTime() - left.lastSeen.getTime())
    .map(user => ({
      email: user.email,
      name: user.email === 'Guest' ? 'Guest' : user.email.split('@')[0],
      lastSeen: user.lastSeen.toISOString(),
    }));
}

function getDashboardData() {
  const ss = SpreadsheetApp.openById(SPREADSHEET_ID);
  const bagSheet = findSheet_(ss, ['bag']);
  const semiSheet = findSheet_(ss, ['semi']);
  const toteSheet = findSheet_(ss, ['tote']);
  const bag = parseBag_(bagSheet);
  const semi = parseRows_(semiSheet, 'semi');
  const tote = parseRows_(toteSheet, 'tote');
  const dhSheet = findSheet_(ss, ['dh name', 'cut-off', 'cutoff', 'dh']);
  const masterDhRows = parseDh_(dhSheet);
  const dhRows = mapRawDestinationsToDh_(masterDhRows, bag, semi, tote);
  const vehicleCaps = parseVehicleCaps_(findSheet_(ss, ['load capacity', 'capacity']));
  const maxVehicles = parseMaxVehicles_(findSheet_(ss, ['vehicle capacity']));
  const loads = computeLoads_(dhRows, bag, semi, tote);
  const rows = buildDhRows_(dhRows, loads, maxVehicles, vehicleCaps);
  const dataUpdated = latestSheetUpdated_([bagSheet, semiSheet, toteSheet]) || new Date();

  return {
    updatedAt: dataUpdated.toISOString(),
    overall: {
      bagCount: bag.length,
      bagShipments: sum_(bag, 'shipments'),
      semiCount: semi.length,
      toteCount: tote.length,
    },
    rows: rows,
    lanes: unique_(masterDhRows.map(row => row.laneType).filter(Boolean)).sort(),
    cutoffs: unique_(masterDhRows.map(row => row.cutoff)).sort(),
    diagnostics: {
      cutoffSheetRows: masterDhRows.length,
      rawDestinationCount: unique_(bag.concat(semi, tote).map(row => normalize_(row.destination))).length,
      matchedDhRows: dhRows.length,
      loadRows: rows.length,
    },
    vehicleCaps: vehicleCaps.map(item => ({vehicle: item.vehicle, capacity: item.capacity})),
    maxVehicles: maxVehicles,
  };
}

function latestSheetUpdated_(sheets) {
  let latest = null;
  sheets.forEach(sheet => {
    if (!sheet) return;
    const table = table_(sheet);
    const updatedHeader = column_(table.headers, header =>
      header.includes('last updated') || header.includes('updated at')
    );
    if (!updatedHeader) return;
    table.rows.forEach(row => {
      const parsed = parseSheetDate_(row[updatedHeader]);
      if (parsed && (!latest || parsed.getTime() > latest.getTime())) latest = parsed;
    });
  });
  return latest;
}

function parseSheetDate_(value) {
  const text = clean_(value);
  if (!text) return null;
  for (const format of ['dd MMM yyyy, hh:mm a', 'd MMM yyyy, hh:mm a']) {
    try {
      const parsed = Utilities.parseDate(text, Session.getScriptTimeZone(), format);
      if (parsed && !isNaN(parsed.getTime())) return parsed;
    } catch (error) {
      // Try the next supported display format.
    }
  }
  const fallback = new Date(text);
  return isNaN(fallback.getTime()) ? null : fallback;
}

function findSheet_(ss, hints) {
  const sheets = ss.getSheets();
  const names = sheets.map(sheet => sheet.getName());
  for (const hint of hints) {
    const exact = sheets.find(sheet => sheet.getName().trim().toLowerCase() === hint);
    if (exact) return exact;
  }
  for (const hint of hints) {
    const match = sheets.find(sheet => sheet.getName().toLowerCase().includes(hint));
    if (match) return match;
  }
  return null;
}

function table_(sheet, headerRow) {
  if (!sheet) return {headers: [], rows: []};
  const values = sheet.getDataRange().getDisplayValues();
  const start = headerRow || 0;
  if (values.length <= start) return {headers: [], rows: []};
  const headers = values[start].map(value => String(value).trim());
  const rows = values.slice(start + 1).map(valuesRow => {
    const row = {};
    headers.forEach((header, index) => row[header] = valuesRow[index] || '');
    return row;
  });
  return {headers: headers, rows: rows};
}

function column_(headers, predicate) {
  return headers.find(header => predicate(String(header).toLowerCase())) || null;
}

function parseBag_(sheet) {
  const table = table_(sheet);
  const destination = column_(table.headers, header => header.includes('dest'));
  const tracking = column_(table.headers, header => header.includes('tracking') && header.includes('count'));
  if (!destination || !tracking) return [];
  return table.rows
    .map(row => ({
      destination: clean_(row[destination]),
      shipments: number_(row[tracking]),
    }))
    .filter(row => row.destination);
}

function parseRows_(sheet, type) {
  const table = table_(sheet);
  const destination = column_(table.headers, header => header.includes('dest'));
  if (!destination) return [];
  return table.rows
    .map(row => ({destination: clean_(row[destination]), type: type}))
    .filter(row => row.destination);
}

function parseDh_(sheet) {
  if (!sheet) return [];
  const values = sheet.getDataRange().getDisplayValues();
  let headerRow = 0;
  for (let index = 0; index < Math.min(values.length, 6); index += 1) {
    const text = values[index].join(' ').toLowerCase();
    if (text.includes('dh') || text.includes('cutoff') || text.includes('cut-off') || text.includes('shift')) {
      headerRow = index;
      break;
    }
  }
  const headers = values[headerRow].map(value => String(value).trim());
  const namedCutoffHeader = headers.find(header => /cut[\s-]*off|cutoff|time/i.test(header));
  const cutoffHeader = namedCutoffHeader || headers.find(header => {
    const index = headers.indexOf(header);
    return values.slice(headerRow + 1, headerRow + 31)
      .filter(row => /^\d{1,2}:\d{2}/.test(String(row[index] || '').trim())).length >= 3;
  });
  if (!cutoffHeader) return [];

  const nameHeader = column_(headers, header =>
    header === 'dh name' || header.includes('dh name') || header === 'delivery hub'
  );
  const codeHeader = column_(headers, header =>
    header.includes('dh code') || header === 'code' || header === 'coc'
  );
  const laneHeader = column_(headers, header =>
    header === 'lane type' || header.includes('lane type')
  );
  if (!nameHeader) return [];
  const rows = values.slice(headerRow + 1).map(valuesRow => {
    const row = {};
    headers.forEach((header, index) => row[header] = valuesRow[index] || '');
    return row;
  });
  return rows
    .map(row => ({
      code: clean_(codeHeader ? row[codeHeader] : ''),
      name: clean_(row[nameHeader]),
      cutoff: clean_(row[cutoffHeader]).slice(0, 5),
      laneType: clean_(laneHeader ? row[laneHeader] : ''),
    }))
    .filter(row => row.name && /^\d{1,2}:\d{2}/.test(row.cutoff));
}

function parseVehicleCaps_(sheet) {
  const fallback = Object.keys(VEHICLE_CFT)
    .map(vehicle => ({vehicle: vehicle, capacity: VEHICLE_CFT[vehicle]}));
  if (!sheet) return fallback;
  const table = table_(sheet);
  const parsed = [];
  const seen = {};
  table.rows.forEach(row => {
    const firstValue = table.headers.length ? row[table.headers[0]] : '';
    const size = vehicleNumber_(firstValue);
    if (size === null || seen[size]) return;
    seen[size] = true;
    const vehicle = vehicleLabel_(size);
    parsed.push({vehicle: vehicle, capacity: VEHICLE_CFT[vehicle] || size * 8 * 9});
  });
  return parsed.length ? parsed.sort((a, b) => a.capacity - b.capacity) : fallback;
}

function parseMaxVehicles_(sheet) {
  if (!sheet) return {};
  const table = table_(sheet);
  const nameHeader = table.headers.find(header => header.trim().toLowerCase() === 'dh name');
  const sizeHeader = table.headers.find(header => header.trim().toLowerCase() === 'vehicle size');
  if (!nameHeader || !sizeHeader) return {};
  const result = {};
  table.rows.forEach(row => {
    const name = clean_(row[nameHeader]);
    const size = clean_(row[sizeHeader]);
    if (name && size) result[normalize_(name)] = size;
  });
  return result;
}

function mapRawDestinationsToDh_(dhRows, bag, semi, tote) {
  const rawNames = {};
  bag.concat(semi, tote).forEach(row => {
    const name = clean_(row.destination);
    if (name) rawNames[normalize_(name)] = name;
  });
  const master = {};
  dhRows.forEach(row => {
    if (!master[normalize_(row.name)]) master[normalize_(row.name)] = row;
  });
  return Object.keys(rawNames)
    .filter(key => master[key])
    .map(key => Object.assign({}, master[key], {name: rawNames[key]}));
}

function computeLoads_(dhRows, bag, semi, tote) {
  const bagAgg = aggregate_(bag, true);
  const semiAgg = aggregate_(semi, false);
  const toteAgg = aggregate_(tote, false);
  const result = {};
  dhRows.forEach(row => {
    const key = normalize_(row.name);
    result[row.name] = {
      bagCount: valueAt_(bagAgg, key, 0),
      bagShipments: valueAt_(bagAgg, key, 1),
      semiCount: valueAt_(semiAgg, key, 0),
      toteCount: valueAt_(toteAgg, key, 0),
    };
  });
  return result;
}

function aggregate_(rows, withShipments) {
  const result = {};
  rows.forEach(row => {
    const key = normalize_(row.destination);
    if (!result[key]) result[key] = [0, 0];
    result[key][0] += 1;
    if (withShipments) result[key][1] += row.shipments || 0;
  });
  return result;
}

function buildDhRows_(dhRows, loads, maxVehicles, vehicleCaps) {
  return dhRows.map(row => {
    const load = loads[row.name] || {bagCount: 0, bagShipments: 0, semiCount: 0, toteCount: 0};
    const totalShipments = load.bagShipments + load.semiCount + load.toteCount;
    const loadCft = loadToCft_(load);
    const maxVehicle = maxVehicles[normalize_(row.name)] || '24 Ft';
    const maxCapacity = capacityForVehicle_(maxVehicle);
    const plan = recommendConstrained_(loadCft, vehicleCaps, maxVehicle);
    return {
      cutoff: row.cutoff,
      code: row.code,
      name: row.name,
      laneType: row.laneType || '',
      bagCount: load.bagCount,
      bagShipments: load.bagShipments,
      semiCount: load.semiCount,
      toteCount: load.toteCount,
      totalShipments: totalShipments,
      maxVehicle: maxVehicle,
      maxUtilization: maxCapacity ? (loadCft / maxCapacity) * 100 : 0,
      recommendedVehicle: plan.vehicle || '—',
      recommendedUtilization: plan.utilization * 100,
      loadCft: loadCft,
      breakdown: plan.breakdown,
    };
  }).filter(row => row.totalShipments > 0)
    .sort((a, b) => a.cutoff.localeCompare(b.cutoff) || a.name.localeCompare(b.name));
}

function recommendConstrained_(loadCft, vehicleCaps, maxVehicle) {
  if (loadCft <= 0) return emptyPlan_();
  const allowed = maxVehicle
    ? vehicleCaps.filter(item => vehicleNumber_(item.vehicle) <= vehicleNumber_(maxVehicle) + 1e-6)
    : vehicleCaps;
  return recommend_(loadCft, allowed.length ? allowed : vehicleCaps);
}

function recommend_(loadCft, vehicleCaps) {
  if (loadCft <= 0 || !vehicleCaps.length) return emptyPlan_();
  const caps = vehicleCaps.slice().sort((a, b) => a.capacity - b.capacity);
  const largest = caps[caps.length - 1];
  if (loadCft <= largest.capacity) {
    const vehicle = caps.find(item => item.capacity >= loadCft) || largest;
    return {
      vehicle: vehicle.vehicle,
      capacity: vehicle.capacity,
      utilization: loadCft / vehicle.capacity,
      breakdown: [{vehicle: vehicle.vehicle, capacity: vehicle.capacity, utilization: loadCft / vehicle.capacity}],
    };
  }
  let remaining = loadCft;
  const breakdown = [];
  while (remaining > largest.capacity) {
    breakdown.push({vehicle: largest.vehicle, capacity: largest.capacity, utilization: 1});
    remaining -= largest.capacity;
  }
  const tail = caps.find(item => item.capacity >= remaining) || largest;
  breakdown.push({vehicle: tail.vehicle, capacity: tail.capacity, utilization: remaining / tail.capacity});
  return {
    vehicle: formatBreakdown_(breakdown),
    capacity: tail.capacity,
    utilization: breakdown[breakdown.length - 1].utilization,
    breakdown: breakdown,
  };
}

function emptyPlan_() {
  return {vehicle: '', capacity: 0, utilization: 0, breakdown: []};
}

function formatBreakdown_(breakdown) {
  const parts = [];
  let index = 0;
  while (index < breakdown.length) {
    const vehicle = breakdown[index].vehicle;
    let count = 1;
    while (index + count < breakdown.length && breakdown[index + count].vehicle === vehicle) count += 1;
    parts.push(count > 1 ? vehicle + ' × ' + count : vehicle);
    index += count;
  }
  return parts.join(' + ');
}

function loadToCft_(load) {
  return (load.bagShipments / SHIPMENTS_PER_BAG) * CFT_PER_BAG
    + load.semiCount * CFT_PER_SEMI
    + load.toteCount * CFT_PER_TOTE;
}

function vehicleLabel_(size) {
  return Object.keys(VEHICLE_CFT).find(vehicle => vehicleNumber_(vehicle) === size) || size + ' Ft';
}

function capacityForVehicle_(value) {
  const size = vehicleNumber_(value);
  if (size === null) return 0;
  const label = vehicleLabel_(size);
  return VEHICLE_CFT[label] || size * 8 * 9;
}

function vehicleNumber_(value) {
  const match = String(value || '').match(/(\d+(?:\.\d+)?)/);
  return match ? Number(match[1]) : null;
}

function normalize_(value) {
  return String(value || '').toLowerCase().replace(/[_\s-]+/g, '');
}

function clean_(value) {
  const text = String(value || '').trim();
  return text && text.toLowerCase() !== 'nan' ? text : '';
}

function number_(value) {
  const parsed = Number(String(value || '').replace(/,/g, '').trim());
  return isNaN(parsed) ? 0 : parsed;
}

function valueAt_(object, key, index) {
  return object[key] ? object[key][index] : 0;
}

function sum_(rows, key) {
  return rows.reduce((total, row) => total + (Number(row[key]) || 0), 0);
}

function unique_(values) {
  return Array.from(new Set(values));
}
