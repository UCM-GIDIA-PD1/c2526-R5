// main.js — Metro-style rendering (líneas como trazos + burbujas por parada)

// ============================================================
// 1. Inicializar Mapa
// ============================================================
const map = L.map('map', {
    center: [40.7128, -74.0060],
    zoom: 12,
    zoomControl: false
});

L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
    subdomains: 'abcd',
    maxZoom: 20
}).addTo(map);

// ============================================================
// 2. Paleta oficial MTA
// ============================================================
const ROUTE_COLORS = {
    '1': '#EE352E', '2': '#EE352E', '3': '#EE352E',
    '4': '#00933C', '5': '#00933C', '6': '#00933C',
    '7': '#B933AD',
    'A': '#0039A6', 'C': '#0039A6', 'E': '#0039A6',
    'B': '#FF6319', 'D': '#FF6319', 'F': '#FF6319', 'M': '#FF6319',
    'G': '#6CBE45',
    'J': '#996633', 'Z': '#996633',
    'L': '#A7A9AC',
    'N': '#FCCC0A', 'Q': '#FCCC0A', 'R': '#FCCC0A', 'W': '#FCCC0A',
    'S': '#808183', 'SIR': '#0039A6'
};

// Offset en píxeles para separar líneas que comparten paradas
const LINE_OFFSET_PX = 4;

function getStationColor(routes) {
    if (!routes) return '#3B82F6';
    const primaryRoute = routes.split(' ')[0];
    return ROUTE_COLORS[primaryRoute] || '#3B82F6';
}

// ============================================================
// 3. Estado global
// ============================================================
let markersMap = {};
let allStations = [];
let currentAlerts = [];    // AlertPrediction[] del WebSocket (por route_id)
let routePolylines = {};   // lineCode -> L.polyline
let stationRouteIndex = {}; // stationId -> [lineCode, lineCode, ...]

const SUBWAY_STATIONS_URL = '/api/stations';
const SUBWAY_ROUTES_URL   = '/api/routes';  // Nuevo endpoint: devuelve { lineCode: [stationId, ...] }

// ============================================================
// 4. Helpers de offset visual (para separar líneas paralelas)
// ============================================================

/**
 * Aplica un offset perpendicular en píxeles a una polilínea
 * para que dos líneas compartiendo paradas no se solapen.
 */
function offsetLatLngs(latlngs, offsetPx) {
    if (offsetPx === 0 || latlngs.length < 2) return latlngs;
    return latlngs.map((pt, i) => {
        const prev = latlngs[Math.max(0, i - 1)];
        const next = latlngs[Math.min(latlngs.length - 1, i + 1)];
        // Vector dirección del segmento (en píxeles de mapa)
        const p1 = map.latLngToLayerPoint(prev);
        const p2 = map.latLngToLayerPoint(next);
        const dx = p2.x - p1.x;
        const dy = p2.y - p1.y;
        const len = Math.sqrt(dx * dx + dy * dy) || 1;
        // Perpendicular normalizado
        const nx = -dy / len;
        const ny =  dx / len;
        // Punto original en píxeles
        const ppt = map.latLngToLayerPoint(pt);
        const shifted = L.point(ppt.x + nx * offsetPx, ppt.y + ny * offsetPx);
        return map.layerPointToLatLng(shifted);
    });
}

// ============================================================
// 5. Crear marcador SVG estilo metro con burbujas por línea
// ============================================================

/**
 * Genera un L.divIcon con una o varias burbujas coloreadas
 * (una por línea que pasa por la parada), alineadas horizontalmente.
 */
function createMetroMarker(routes) {
    const routeList = routes ? routes.split(' ') : [];
    const R = 7;          // radio de burbuja
    const GAP = 2;        // separación entre burbujas
    const n = routeList.length;
    const totalW = n * (R * 2) + (n - 1) * GAP;
    const totalH = R * 2 + 4; // algo de padding vertical

    const circles = routeList.map((route, i) => {
        const color = ROUTE_COLORS[route] || '#3B82F6';
        const cx = i * (R * 2 + GAP) + R;
        const cy = R + 2;
        return `<circle cx="${cx}" cy="${cy}" r="${R}"
                    fill="${color}"
                    stroke="white" stroke-width="1.5"
                    style="filter:drop-shadow(0 0 3px ${color}88)"/>
                <text x="${cx}" y="${cy}"
                    dominant-baseline="central" text-anchor="middle"
                    font-family="Arial,sans-serif" font-size="${R < 7 ? 6 : 7}"
                    font-weight="bold" fill="white">${route.length <= 2 ? route : ''}</text>`;
    }).join('');

    const svg = `<svg xmlns="http://www.w3.org/2000/svg"
                    width="${totalW}" height="${totalH}"
                    viewBox="0 0 ${totalW} ${totalH}"
                    style="overflow:visible">
                    ${circles}
                </svg>`;

    return L.divIcon({
        className: 'metro-marker',
        html: svg,
        iconSize: [totalW, totalH],
        iconAnchor: [totalW / 2, totalH / 2]
    });
}

// ============================================================
// 6. Dibujar líneas de metro como polilíneas
// ============================================================

/**
 * Dibuja las polilíneas de cada línea de metro.
 * Requiere que el backend devuelva /api/routes:
 *   { "1": ["sta_id_1", "sta_id_2", ...], "A": [...], ... }
 */
function haversineKm(a, b) {
    const R = 6371;
    const dLat = (b.lat - a.lat) * Math.PI / 180;
    const dLon = (b.lon - a.lon) * Math.PI / 180;
    const s = Math.sin(dLat/2) * Math.sin(dLat/2) +
              Math.cos(a.lat * Math.PI/180) * Math.cos(b.lat * Math.PI/180) *
              Math.sin(dLon/2) * Math.sin(dLon/2);
    return R * 2 * Math.atan2(Math.sqrt(s), Math.sqrt(1-s));
}

/**
 * Ordena las estaciones de una línea usando el algoritmo de vecino más cercano
 * empezando desde la estación más al norte. Esto evita los zigzags.
 */
function sortByNearestNeighbor(ids, stationsById) {
    if (ids.length <= 2) return ids;
    const remaining = [...ids];
    // Empezar desde la estación más al norte
    remaining.sort((a, b) => stationsById[b].lat - stationsById[a].lat);
    const ordered = [remaining.shift()];

    while (remaining.length > 0) {
        const last = stationsById[ordered[ordered.length - 1]];
        let bestIdx = 0;
        let bestDist = Infinity;
        remaining.forEach((id, i) => {
            const dist = haversineKm(last, stationsById[id]);
            if (dist < bestDist) { bestDist = dist; bestIdx = i; }
        });
        // Solo conectar si la distancia es razonable (< 3 km entre paradas)
        if (bestDist > 3) break;
        ordered.push(remaining.splice(bestIdx, 1)[0]);
    }
    return ordered;
}

function drawRouteLines(routeData, stationsById) {
    const lineList = Object.keys(routeData);

    lineList.forEach((lineCode, lineIndex) => {
        const rawIds = routeData[lineCode];
        // Ordenar por vecino más cercano para evitar zigzags
        const stationIds = sortByNearestNeighbor(rawIds, stationsById);

        const latlngs = stationIds
            .map(id => stationsById[id])
            .filter(Boolean)
            .map(st => [st.lat, st.lon]);

        if (latlngs.length < 2) return;

        const color = ROUTE_COLORS[lineCode] || '#3B82F6';

        // Borde oscuro debajo para dar profundidad
        L.polyline(latlngs, {
            color: darkenColor(color, 0.4),
            weight: 7,
            opacity: 0.6,
            lineJoin: 'round',
            lineCap: 'round'
        }).addTo(map);

        // Línea principal
        const polyline = L.polyline(latlngs, {
            color: color,
            weight: 4,
            opacity: 0.9,
            lineJoin: 'round',
            lineCap: 'round',
            className: `metro-line metro-line-${lineCode}`
        }).addTo(map);

        routePolylines[lineCode] = polyline;
    });
}

function darkenColor(hex, factor) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgb(${Math.round(r * factor)},${Math.round(g * factor)},${Math.round(b * factor)})`;
}

// ============================================================
// 7. Carga de datos y renderizado
// ============================================================

Promise.all([
    fetch(SUBWAY_STATIONS_URL).then(r => r.json()),
    fetch('/api/routes').then(r => r.json()).catch(() => ({}))
]).then(([stations, routeData]) => {
    allStations = stations;

    const stationsById = {};
    stations.forEach(st => {
        stationsById[st.id] = st;
        stationRouteIndex[st.id] = st.routes ? st.routes.split(' ') : [];
    });

    // Dibujar líneas con orden oficial del backend
    drawRouteLines(routeData, stationsById);

    // Dibujar marcadores encima
    stations.forEach(station => {
        const icon = createMetroMarker(station.routes);
        const marker = L.marker([station.lat, station.lon], {
            icon,
            zIndexOffset: 1000
        });
        marker.on('click', () => openStationDetails(station));
        markersMap[station.id] = marker.addTo(map);
    });

    console.log(`${stations.length} estaciones renderizadas con estilo metro.`);
    initRoutePlanner();
}).catch(err => console.error("Error al cargar datos:", err));

// ============================================================
// 8. WebSocket para predicciones en tiempo real
// ============================================================
const wsUrl = `ws://${window.location.host}/ws/live-updates`;
const socket = new WebSocket(wsUrl);

socket.onopen  = () => console.log("Conectado a FastAPI WebSocket");
socket.onclose = () => console.log("WebSocket desconectado");

socket.onmessage = (event) => {
    const payload = JSON.parse(event.data);
    if (payload.type === 'update' || payload.type === 'initial') {
        currentAlerts = payload.alerts?.predictions || [];
        updateMapWithAlerts(currentAlerts);
        if (!detailPanel.classList.contains('hidden')) {
            checkActiveAlerts();
        }
    }
};

// ============================================================
// 9. Actualización de marcadores con alertas
// ============================================================

function renderDelayCard(id, delaySeconds) {
    const el = document.getElementById(id);
    if (!el) return;
    if (delaySeconds === null || delaySeconds === undefined) {
        el.textContent = '--';
        el.className = 'delay-val';
        return;
    }
    const s = Math.round(delaySeconds);
    let text;
    if (s < 60) {
        text = `${s} s`;
    } else {
        const m = Math.floor(s / 60);
        const r = s % 60;
        text = r > 0 ? `${m} min ${r} s` : `${m} min`;
    }
    const cls = s <= 30 ? 'on-time' : (s < 120 ? 'minor-delay' : 'delayed');
    el.textContent = text;
    el.className = `delay-val ${cls}`;
}

function updateMapWithAlerts(alertPredictions) {
    // Conjunto de route_ids con alerta activa
    const alertedRoutes = new Set(
        alertPredictions.filter(a => a.alert_predicted).map(a => a.route_id)
    );

    Object.entries(markersMap).forEach(([stationId, marker]) => {
        const routes = stationRouteIndex[stationId] || [];
        const hasAlert = routes.some(r => alertedRoutes.has(r));
        const el = marker.getElement();
        if (el) el.classList.toggle('metro-delayed', hasAlert);
    });
}

// ============================================================
// 10. Panel de detalles de estación
// ============================================================

const detailPanel  = document.getElementById('station-detail-panel');
const closeDetailBtn = document.getElementById('close-detail');
closeDetailBtn.onclick = () => detailPanel.classList.add('hidden');

function openStationDetails(station) {
    detailPanel.currentStation = station;
    const nameEl        = document.getElementById('detail-station-name');
    const lineSelectors = document.getElementById('detail-line-selectors');

    detailPanel.classList.remove('hidden');
    map.flyTo([station.lat, station.lon], 14, { duration: 1.5 });

    nameEl.textContent   = station.name;
    lineSelectors.innerHTML = '';

    const routes = station.routes.split(' ');
    routes.forEach((route, index) => {
        const bubble = document.createElement('div');
        const color  = ROUTE_COLORS[route] || '#3B82F6';
        bubble.className   = `line-bubble ${index === 0 ? 'active' : ''}`;
        bubble.textContent = route;
        bubble.style.backgroundColor = color;
        bubble.onclick = () => {
            document.querySelectorAll('.line-bubble').forEach(b => b.classList.remove('active'));
            bubble.classList.add('active');
            updateForecast(route);
        };
        lineSelectors.appendChild(bubble);
    });

    if (routes.length > 0) updateForecast(routes[0]);
}

async function updateForecast(lineCode) {
    detailPanel.currentLine = lineCode;
    const station = detailPanel.currentStation;

    ['forecast-now', 'forecast-10', 'forecast-20', 'forecast-30'].forEach(id => {
        const el = document.getElementById(id);
        el.textContent = '…';
        el.className = 'delay-val';
    });

    const stopParam = encodeURIComponent(station.id);

    // Lanzar las tres peticiones en paralelo
    const [nowResp, propResp, alertResp] = await Promise.allSettled([
        fetch(`/api/predict/current?stop_id=${stopParam}`),
        fetch(`/api/predict/propagation?stop_id=${stopParam}`),
        fetch(`/api/predict/alerts?route_id=${encodeURIComponent(lineCode)}`),
    ]);

    // Ahora: último delay observado en la ventana más reciente (en segundos)
    try {
        if (nowResp.status === 'fulfilled' && nowResp.value.ok) {
            const d = await nowResp.value.json();
            renderDelayCard('forecast-now', d.delay_seconds);
        } else {
            renderDelayCard('forecast-now', null);
        }
    } catch { renderDelayCard('forecast-now', null); }

    // +10/+20/+30 min: predicción DCRNN (valores en segundos)
    try {
        if (propResp.status === 'fulfilled' && propResp.value.ok) {
            const d = await propResp.value.json();
            const pred = d.predictions?.find(p => p.stop_id === station.id);
            if (pred) {
                renderDelayCard('forecast-10', pred.delay_10m);
                renderDelayCard('forecast-20', pred.delay_20m);
                renderDelayCard('forecast-30', pred.delay_30m);
            } else {
                ['forecast-10', 'forecast-20', 'forecast-30'].forEach(id => renderDelayCard(id, null));
            }
        } else {
            ['forecast-10', 'forecast-20', 'forecast-30'].forEach(id => renderDelayCard(id, null));
        }
    } catch { ['forecast-10', 'forecast-20', 'forecast-30'].forEach(id => renderDelayCard(id, null)); }

    // Probabilidad de alerta para la línea seleccionada
    try {
        if (alertResp.status === 'fulfilled' && alertResp.value.ok) {
            const d = await alertResp.value.json();
            renderAlertProbability(lineCode, d.predictions);
        } else {
            renderAlertProbability(lineCode, []);
        }
    } catch { renderAlertProbability(lineCode, []); }

    document.getElementById('line-detail-info').innerHTML =
        `<p>Línea <strong>${lineCode}</strong> · predicción DCRNN a 10, 20 y 30 min</p>`;
}

function renderAlertProbability(lineCode, predictions) {
    const section = document.getElementById('station-alert');
    const textEl  = section.querySelector('.alert-text');
    const iconEl  = section.querySelector('.alert-icon');

    if (!predictions || predictions.length === 0) {
        section.classList.add('hidden');
        return;
    }

    // Tomar la probabilidad máxima entre todas las direcciones de la línea
    const maxProb = Math.max(...predictions.map(p => p.alert_probability));
    const pct = Math.round(maxProb * 100);
    const predicted = predictions.some(p => p.alert_predicted);

    section.classList.remove('hidden');
    if (predicted) {
        iconEl.textContent = '⚠️';
        textEl.textContent = `Alerta probable en línea ${lineCode} (${pct}%)`;
        section.querySelector('.alert-banner').style.background = 'rgba(239,68,68,0.15)';
    } else {
        iconEl.textContent = '✓';
        textEl.textContent = `Sin alerta en línea ${lineCode} (${pct}% prob.)`;
        section.querySelector('.alert-banner').style.background = 'rgba(34,197,94,0.1)';
    }
}

function checkActiveAlerts() {
    const alertSection = document.getElementById('station-alert');
    if (!detailPanel.currentStation || !detailPanel.currentLine) return;
    const stationRoutes = stationRouteIndex[detailPanel.currentStation.id] || [];
    const hasAlert = currentAlerts.some(a => a.alert_predicted && stationRoutes.includes(a.route_id));
    alertSection.classList.toggle('hidden', !hasAlert);
}

// ============================================================
// 11. Planificador de rutas con autocompletado
// ============================================================

function initRoutePlanner() {
    setupAutocomplete(
        document.getElementById('origin-input'),
        document.getElementById('origin-results')
    );
    setupAutocomplete(
        document.getElementById('destination-input'),
        document.getElementById('destination-results')
    );
}

function setupAutocomplete(input, dropdown) {
    input.addEventListener('input', () => {
        const value = input.value.toLowerCase().trim();
        if (value.length < 2) { dropdown.classList.add('hidden'); return; }

        const filtered = allStations.filter(s => s.name.toLowerCase().includes(value)).slice(0, 5);

        if (filtered.length > 0) {
            dropdown.innerHTML = '';
            filtered.forEach(s => {
                const item = document.createElement('div');
                item.className   = 'suggestion-item';
                item.textContent = s.name;
                item.onclick = () => {
                    input.value = s.name;
                    input.dataset.stationId = s.id;
                    dropdown.classList.add('hidden');
                    calculateRoute();
                };
                dropdown.appendChild(item);
            });
            dropdown.classList.remove('hidden');
        } else {
            dropdown.classList.add('hidden');
        }
    });

    document.addEventListener('click', (e) => {
        if (!input.contains(e.target) && !dropdown.contains(e.target))
            dropdown.classList.add('hidden');
    });
}

async function calculateRoute() {
    const originId  = document.getElementById('origin-input').dataset.stationId;
    const destId    = document.getElementById('destination-input').dataset.stationId;
    const resultPanel = document.getElementById('route-prediction');

    if (!originId || !destId) return;

    const origin      = allStations.find(s => s.id === originId);
    const destination = allStations.find(s => s.id === destId);

    resultPanel.classList.remove('hidden');

    const commonLines = findCommonLines(origin, destination);
    const displayLine = commonLines.length > 0 ? commonLines[0] : origin.routes.split(' ')[0];

    document.getElementById('result-line').textContent = displayLine;
    document.getElementById('result-line').style.backgroundColor = ROUTE_COLORS[displayLine] || '#3B82F6';
    document.getElementById('result-station').textContent = origin.name;

    renderDelayCard('result-delay-val', null);

    try {
        const resp = await fetch(`/api/predict/delay/30m?stop_id=${encodeURIComponent(originId)}`);
        if (!resp.ok) throw new Error(resp.statusText);
        const data = await resp.json();
        const pred = data.predictions?.[0];
        renderDelayCard('result-delay-val', pred ? pred.delay_minutes * 60 : null);
    } catch (e) {
        console.error('Error al obtener retraso de ruta:', e);
    }
}

function findCommonLines(s1, s2) {
    const r1 = s1.routes.split(' ');
    const r2 = s2.routes.split(' ');
    return r1.filter(line => r2.includes(line));
}