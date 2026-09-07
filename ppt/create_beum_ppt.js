const pptxgen = require('/Users/mac/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/pptxgenjs');
const fs = require('fs');
const path = require('path');
const { Document, Packer, Paragraph, HeadingLevel, AlignmentType } = require('/Users/mac/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/docx');

const OUT = '/Users/mac/project/BEUM/ppt';
const MODEL_VIEW = path.join(OUT, 'beum_3d_model_assembly_view.png');
const MODEL_EXPLODED = path.join(OUT, 'beum_3d_model_exploded.png');
const PPTX = path.join(OUT, '비움_하드웨어_양산로드맵.pptx');
const MD = path.join(OUT, '비움_하드웨어_발표대본.md');
const DOCX = path.join(OUT, '비움_하드웨어_발표대본.docx');

const C = {
  navy: '0D2138', navy2: '153451', ink: '1A2636', sub: '66758A', muted: '9AA7B7',
  bg: 'F4F7FA', white: 'FFFFFF', cream: 'FFFDF1', blue: '2B77E5', cyan: '12B6A4',
  amber: 'F2B544', coral: 'E66964', line: 'DCE4ED', paleBlue: 'E8F1FF',
  paleCyan: 'E5F8F5', paleAmber: 'FFF5D9', paleCoral: 'FFF0EF'
};
const FONT = 'Apple SD Gothic Neo';

const pptx = new pptxgen();
pptx.defineLayout({ name: 'BEUM_WIDE', width: 13.333, height: 7.5 });
pptx.layout = 'BEUM_WIDE';
pptx.author = 'BEUM';
pptx.company = 'BEUM';
pptx.subject = '비움 하드웨어 기획 의도, 현실적 한계, 고도화 로드맵';
pptx.title = '비움 하드웨어: 시연형에서 제품형으로';
pptx.lang = 'ko-KR';
pptx.theme = { headFontFace: FONT, bodyFontFace: FONT, lang: 'ko-KR' };

function text(s, v, x, y, w, h, o = {}) {
  s.addText(v, { x, y, w, h, fontFace: FONT, lang: 'ko-KR', color: C.ink, margin: 0, fit: 'shrink', valign: 'mid', ...o });
}
function round(s, x, y, w, h, fill, border = fill, pt = 1) {
  s.addShape(pptx.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.1, fill: { color: fill }, line: { color: border, pt, transparency: border === fill ? 100 : 0 } });
}
function line(s, x, y, w, h, color, pt = 1, arrow = false) {
  s.addShape(pptx.ShapeType.line, { x, y, w, h, line: { color, pt, endArrowType: arrow ? 'triangle' : undefined } });
}
function dot(s, x, y, size, fill) {
  s.addShape(pptx.ShapeType.ellipse, { x, y, w: size, h: size, fill: { color: fill }, line: { color: fill, transparency: 100 } });
}
function eyebrow(s, value, page, dark = false) {
  text(s, value.toUpperCase(), 0.62, 0.35, 4.2, 0.18, { fontSize: 9, bold: true, charSpacing: 1.3, color: dark ? '9DB7D4' : C.blue });
  text(s, `BEUM  /  ${page}`, 11.35, 0.35, 1.35, 0.18, { fontSize: 8.5, color: dark ? '9DB7D4' : C.muted, align: 'right' });
}
function pill(s, v, x, y, w, fill, color) {
  round(s, x, y, w, 0.27, fill, fill);
  text(s, v, x + 0.05, y + 0.03, w - 0.10, 0.16, { fontSize: 8.8, bold: true, color, align: 'center' });
}
function iconCamera(s, x, y, fill) {
  round(s, x + 0.03, y + 0.09, 0.42, 0.28, fill, fill);
  dot(s, x + 0.18, y + 0.15, 0.14, C.white);
  line(s, x + 0.45, y + 0.16, 0.10, 0, fill, 3);
}
function iconChip(s, x, y, fill) {
  s.addShape(pptx.ShapeType.rect, { x: x + 0.10, y: y + 0.07, w: 0.33, h: 0.33, fill: { color: fill }, line: { color: fill, transparency: 100 } });
  for (let i = 0; i < 3; i++) { line(s, x + 0.04, y + 0.13 + i * 0.10, 0.06, 0, fill, 1); line(s, x + 0.43, y + 0.13 + i * 0.10, 0.06, 0, fill, 1); }
}
function iconPin(s, x, y, fill) {
  dot(s, x + 0.13, y + 0.06, 0.25, fill); dot(s, x + 0.21, y + 0.14, 0.09, C.white); line(s, x + 0.255, y + 0.29, 0, 0.16, fill, 2.5);
}
function iconServer(s, x, y, fill) {
  round(s, x + 0.06, y + 0.08, 0.39, 0.12, fill, fill); round(s, x + 0.06, y + 0.27, 0.39, 0.12, fill, fill); dot(s, x + 0.12, y + 0.11, 0.05, C.white); dot(s, x + 0.12, y + 0.30, 0.05, C.white);
}
function flowNode(s, x, y, fill, accent, label, sub, icon) {
  round(s, x, y, 1.10, 1.13, fill, fill);
  if (icon === 'camera') iconCamera(s, x + 0.27, y + 0.20, accent);
  if (icon === 'chip') iconChip(s, x + 0.27, y + 0.20, accent);
  if (icon === 'pin') iconPin(s, x + 0.27, y + 0.20, accent);
  if (icon === 'server') iconServer(s, x + 0.27, y + 0.20, accent);
  text(s, label, x + 0.06, y + 0.68, 0.98, 0.18, { fontSize: 10.5, bold: true, color: C.navy, align: 'center' });
  text(s, sub, x + 0.05, y + 0.89, 1.00, 0.13, { fontSize: 7.7, color: C.sub, align: 'center' });
}

// 1. Current rationale and limits
{
  const s = pptx.addSlide(); s.background = { color: C.navy }; eyebrow(s, 'Hardware strategy', '01', true);
  for (const [x, y, r, fill] of [[11.8, -0.4, 1.8, C.navy2], [10.7, 6.85, 2.4, C.navy2], [0.05, 6.55, 0.8, '1A456A']]) {
    s.addShape(pptx.ShapeType.ellipse, { x, y, w: r, h: r, fill: { color: fill, transparency: 45 }, line: { color: fill, transparency: 100 } });
  }
  text(s, '시연을 위한 선택,\n현장을 위한 다음 단계', 0.62, 0.98, 6.35, 0.98, { fontSize: 28, bold: true, color: C.white, breakLine: true, valign: 'top' });
  text(s, '비움의 하드웨어는 “완제품”보다\n“가능성 검증”을 먼저 목표로 설계했습니다.', 0.65, 2.18, 5.6, 0.52, { fontSize: 14, color: 'C9D8E8', breakLine: true, valign: 'top' });
  line(s, 0.66, 3.14, 0.58, 0, C.amber, 2); text(s, '빠르게 만들고, 실제로 확인한다.', 1.42, 3.02, 4.2, 0.26, { fontSize: 13, bold: true, color: C.amber });
  text(s, '시연 단계에서 검증한 범위', 0.66, 5.62, 2.1, 0.18, { fontSize: 9.5, bold: true, color: '9DB7D4' });
  text(s, '탐지  →  위치 기록  →  서버 전송', 0.66, 5.95, 5.2, 0.33, { fontSize: 17, bold: true, color: C.white });
  text(s, '범용 부품으로 기능 흐름과 현장 데이터를 먼저 확보했습니다.', 0.66, 6.45, 5.6, 0.22, { fontSize: 9.8, color: '9DB7D4' });

  round(s, 7.05, 0.72, 5.68, 6.08, C.white, C.white);
  text(s, '현재 시연 구성', 7.50, 1.12, 2.0, 0.22, { fontSize: 10, bold: true, color: C.blue });
  text(s, '저비용·빠른 제작·기능 검증', 7.50, 1.43, 4.4, 0.34, { fontSize: 17, bold: true, color: C.navy });
  line(s, 7.50, 2.08, 4.72, 0, C.line, 1);
  flowNode(s, 7.52, 2.50, C.paleBlue, C.blue, '카메라', '영상 수집', 'camera');
  flowNode(s, 8.78, 2.50, C.paleCyan, C.cyan, '라즈베리파이', '엣지 처리', 'chip');
  flowNode(s, 10.04, 2.50, C.paleAmber, C.amber, 'GPS', '위치 기록', 'pin');
  flowNode(s, 11.30, 2.50, C.paleBlue, C.blue, '서버', '이벤트 저장', 'server');
  for (const x of [8.64, 9.90, 11.16]) line(s, x, 3.03, 0.12, 0, C.muted, 1.4, true);
  pill(s, '빠른 제작', 7.55, 4.08, 1.10, C.paleBlue, C.blue); pill(s, '낮은 비용', 8.83, 4.08, 1.10, C.paleCyan, C.cyan); pill(s, '현장 데이터', 10.11, 4.08, 1.22, C.paleAmber, '946A00');
  line(s, 7.50, 4.72, 4.72, 0, C.line, 1); text(s, '현실적 한계', 7.50, 4.98, 1.55, 0.22, { fontSize: 10, bold: true, color: C.coral });
  [['01', '시야 가림', '차량·사람·각도'], ['02', '우천·야간', '화질·반사·물방울'], ['03', '야외 내구성', '전원·방수·통신']].forEach((v, i) => {
    const x = 7.50 + i * 1.54; pill(s, v[0], x, 5.43, 0.36, C.paleCoral, C.coral); text(s, v[1], x, 5.83, 1.36, 0.19, { fontSize: 10, bold: true, color: C.navy }); text(s, v[2], x, 6.12, 1.38, 0.30, { fontSize: 8.2, color: C.sub, valign: 'top' });
  });
  text(s, '현재 선택은 부족함이 아니라, 시연 목표에 맞춘 우선순위입니다.', 7.50, 6.59, 4.7, 0.16, { fontSize: 8.5, italic: true, color: C.muted });
  s.addNotes(`슬라이드 1 대본\n\n비움의 현재 하드웨어는 완제품을 전제로 만든 것이 아니라, 핵심 기능을 빠르게 검증하기 위한 시연형 설계입니다. 라즈베리파이와 범용 카메라를 사용한 이유는 저비용으로 제작할 수 있고, 카메라 영상에서 AI가 빗물받이와 막힘 영역을 판단한 뒤 GPS 위치와 함께 서버로 전송하는 전체 흐름을 빠르게 확인할 수 있기 때문입니다.\n\n이 단계에서는 탐지하고, 위치를 기록하고, 결과를 서버로 전달할 수 있는지를 먼저 검증했습니다. 반면 실제 야외 환경에서는 차량이나 사람에 의한 시야 가림, 우천과 야간의 화질 저하, 전원·방수·통신 문제로 인해 한계가 생길 수 있습니다. 따라서 현재 구조는 완제품 수준을 목표로 한 것이 아니라, 다음 단계의 설계를 위한 데이터와 검증 결과를 확보하기 위한 선택이었다고 설명할 수 있습니다.`);
}

// 2. Actual 3D printed model and roadmap
{
  const s = pptx.addSlide(); s.background = { color: C.bg }; eyebrow(s, 'Next hardware', '02');
  text(s, '현재 3D 프린터 모델링에서\n제품형 하드웨어로', 0.62, 0.82, 7.2, 0.74, { fontSize: 25, bold: true, color: C.navy, breakLine: true, valign: 'top' });
  text(s, '실제 제작한 케이스를 출발점으로, 탐지 안정성과 현장 내구성을 단계적으로 높입니다.', 0.65, 1.72, 8.2, 0.24, { fontSize: 12.5, color: C.sub });
  round(s, 0.60, 2.10, 7.35, 4.86, C.navy, C.navy);
  text(s, 'ACTUAL PROTOTYPE', 0.95, 2.42, 2.3, 0.18, { fontSize: 9, bold: true, charSpacing: 1.0, color: '9DB7D4' });
  text(s, '3D 프린터 케이스 모델링', 0.95, 2.70, 3.8, 0.28, { fontSize: 16, bold: true, color: C.white });
  round(s, 0.92, 3.16, 4.62, 3.05, C.cream, C.cream); s.addImage({ path: MODEL_VIEW, x: 1.02, y: 3.22, w: 4.42, h: 2.92, sizingContain: true });
  round(s, 5.72, 3.16, 1.84, 1.84, C.cream, C.cream); s.addImage({ path: MODEL_EXPLODED, x: 5.79, y: 3.23, w: 1.70, h: 1.70, sizingContain: true });
  pill(s, '조립도', 6.02, 5.10, 0.86, C.paleCyan, C.cyan); text(s, '실제 출력 케이스의\n배치·조립 가능성 확인', 5.75, 5.54, 1.78, 0.44, { fontSize: 9.2, color: 'C9D8E8', breakLine: true, align: 'center', valign: 'top' });
  [['카메라 장착부', C.blue], ['전자부 수납', C.cyan], ['분해·조립 구조', C.amber]].forEach((v, i) => { const x = 0.95 + i * 2.12; dot(s, x, 6.50, 0.08, v[1]); text(s, v[0], x + 0.16, 6.43, 1.68, 0.20, { fontSize: 9.3, bold: true, color: C.white }); });

  round(s, 8.20, 2.10, 4.52, 4.86, C.white, C.line); text(s, '3단계 고도화 로드맵', 8.58, 2.43, 3.35, 0.28, { fontSize: 16, bold: true, color: C.navy }); line(s, 8.61, 2.98, 3.62, 0, C.line, 1);
  const stages = [
    { n: '01', title: '탐지 보완', body: '카메라 각도·사각지대 점검\n수위 센서 연계 및 점검 필요 분리', fill: C.paleBlue, accent: C.blue },
    { n: '02', title: '현장 내구성', body: '출력 케이스를 전용 하우징으로\n전원·통신·watchdog 보완', fill: C.paleCyan, accent: C.cyan },
    { n: '03', title: '운영 검증', body: '장기 실증·유지보수\n제품 인증·양산성 검토', fill: C.paleAmber, accent: '946A00' }
  ];
  stages.forEach((st, i) => { const y = 3.30 + i * 1.03; round(s, 8.58, y, 0.58, 0.58, st.fill, st.fill); text(s, st.n, 8.58, y + 0.16, 0.58, 0.19, { fontSize: 11.5, bold: true, color: st.accent, align: 'center' }); text(s, st.title, 9.40, y + 0.01, 2.70, 0.22, { fontSize: 12.3, bold: true, color: C.navy }); text(s, st.body, 9.40, y + 0.31, 2.85, 0.42, { fontSize: 9.4, color: C.sub, breakLine: true, valign: 'top' }); if (i < stages.length - 1) line(s, 8.87, y + 0.60, 0, 0.43, C.line, 1.2); });
  round(s, 8.58, 6.26, 3.76, 0.42, C.navy, C.navy); text(s, '현재 모델링을 실제 현장형 제품으로 확장', 8.74, 6.38, 3.44, 0.15, { fontSize: 9.8, bold: true, color: C.white, align: 'center' });
  text(s, '탐지 가능한 장치  →  판단 가능한 시스템', 0.65, 7.16, 5.7, 0.18, { fontSize: 10.5, bold: true, color: C.blue }); text(s, '현재 모델링은 기능·장착 가능성 검증의 결과입니다.', 8.20, 7.16, 4.52, 0.18, { fontSize: 8.5, italic: true, color: C.muted, align: 'right' });
  s.addNotes(`슬라이드 2 대본\n\n두 번째 슬라이드는 실제로 제작한 3D 프린터 케이스 모델링입니다. 미래의 완제품을 가정한 이미지가 아니라, 현재 비움 시연 장치에서 카메라와 전자부를 하나의 케이스에 배치하고 실제 장착 가능성을 확인한 결과입니다.\n\n추후에는 세 단계로 고도화합니다. 첫째, 카메라 각도와 사각지대를 점검하고 수위 센서를 연계해 영상이 가려진 상황도 점검 필요 상태로 구분합니다. 둘째, 3D 프린터 출력 케이스를 출발점으로 전용 하우징, 전원 보호, 통신, watchdog을 보완합니다. 셋째, 장기 현장 실증과 유지보수, 인증과 양산성을 검증합니다. 중요한 점은 현재 모델링을 버리는 것이 아니라, 이 시연 결과를 기반으로 실제 현장형 제품으로 단계적으로 발전시키는 것입니다.`);
}

const slide1 = '비움의 현재 하드웨어는 완제품을 전제로 만든 것이 아니라, 핵심 기능을 빠르게 검증하기 위한 시연형 설계입니다. 라즈베리파이와 범용 카메라를 사용한 이유는 저비용으로 제작할 수 있고, 카메라 영상에서 AI가 빗물받이와 막힘 영역을 판단한 뒤 GPS 위치와 함께 서버로 전송하는 전체 흐름을 빠르게 확인할 수 있기 때문입니다.\n\n이 단계에서는 탐지하고, 위치를 기록하고, 결과를 서버로 전달할 수 있는지를 먼저 검증했습니다. 반면 실제 야외 환경에서는 차량이나 사람에 의한 시야 가림, 우천과 야간의 화질 저하, 전원·방수·통신 문제로 인해 한계가 생길 수 있습니다. 따라서 현재 구조는 완제품 수준을 목표로 한 것이 아니라, 다음 단계의 설계를 위한 데이터와 검증 결과를 확보하기 위한 선택이었다고 설명할 수 있습니다.';
const slide2 = '두 번째 슬라이드는 실제로 제작한 3D 프린터 케이스 모델링입니다. 미래의 완제품을 가정한 이미지가 아니라, 현재 비움 시연 장치에서 카메라와 전자부를 하나의 케이스에 배치하고 실제 장착 가능성을 확인한 결과입니다.\n\n추후에는 세 단계로 고도화합니다. 첫째, 카메라 각도와 사각지대를 점검하고 수위 센서를 연계해 영상이 가려진 상황도 점검 필요 상태로 구분합니다. 둘째, 3D 프린터 출력 케이스를 출발점으로 전용 하우징, 전원 보호, 통신, watchdog을 보완합니다. 셋째, 장기 현장 실증과 유지보수, 인증과 양산성을 검증합니다. 중요한 점은 현재 모델링을 버리는 것이 아니라, 이 시연 결과를 기반으로 실제 현장형 제품으로 단계적으로 발전시키는 것입니다.';
fs.writeFileSync(MD, `# 비움 하드웨어 발표 대본\n\n## 슬라이드 1. 시연을 위한 선택, 현장을 위한 다음 단계\n\n${slide1}\n\n## 슬라이드 2. 현재 3D 프린터 모델링에서 제품형 하드웨어로\n\n${slide2}\n`, 'utf8');
const doc = new Document({ sections: [{ properties: {}, children: [
  new Paragraph({ text: '비움 하드웨어 발표 대본', heading: HeadingLevel.TITLE, alignment: AlignmentType.CENTER }),
  new Paragraph({ text: '발표 분량: 약 2~3분', alignment: AlignmentType.CENTER }),
  new Paragraph({ text: '슬라이드 1. 시연을 위한 선택, 현장을 위한 다음 단계', heading: HeadingLevel.HEADING_1 }),
  new Paragraph({ text: slide1 }),
  new Paragraph({ text: '슬라이드 2. 현재 3D 프린터 모델링에서 제품형 하드웨어로', heading: HeadingLevel.HEADING_1 }),
  new Paragraph({ text: slide2 })
] }] });
Promise.all([pptx.writeFile({ fileName: PPTX }), Packer.toBuffer(doc).then(buf => fs.writeFileSync(DOCX, buf))]).then(() => console.log(`Created: ${PPTX}\nCreated: ${MD}\nCreated: ${DOCX}`));
