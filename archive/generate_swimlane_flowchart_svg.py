#!/usr/bin/env python3
"""
Generate a professional technical SWIMLANE FLOWCHART / PROCESS FLOW DIAGRAM
for Project: 'AI-Based Log Anomaly Detection using DeepLog'
Title: 'Post-Review Progress — HDFS Log Analysis & DeepLog'

Key Characteristics:
- True flowchart styling based directly on workflow/deeplog_workflow_diagram.png
- 5 vertical swimlanes with top headers and distinct light pastel backgrounds
- Rounded rectangle process boxes, decision diamonds, start/end nodes
- Directional connecting arrows with clean elbow routing (left-to-right & top-to-bottom)
- Color coded: Green (validated/Normal), Red (Anomaly/warning), Blue (processing), Orange (analysis), Gray (neutral)
- Academic/engineering research style: clean white background, dark navy text, crisp typography
"""

import os
import xml.etree.ElementTree as ET

WORKSPACE_DIR = "/Users/chalermsak/Desktop/DeepLog-master"
SVG_FILE = os.path.join(WORKSPACE_DIR, "post_review_swimlane_flowchart.svg")
HTML_FILE = os.path.join(WORKSPACE_DIR, "post_review_swimlane_flowchart.html")

def generate_svg():
    width = 1920
    height = 1080
    
    # Palette
    bg_canvas = "#ffffff"
    text_dark = "#0f172a"
    text_muted = "#64748b"
    text_white = "#ffffff"
    border_subtle = "#cbd5e1"
    
    # Lane Header Colors
    lane_header_bg = "#1e293b"
    lane_header_text = "#f8fafc"
    
    # Lane Backgrounds (subtle distinctive tints)
    lane_bgs = [
        "#f8fafc",  # Lane 1: Slate
        "#f0fdf4",  # Lane 2: Light green/mint
        "#fffbeb",  # Lane 3: Warm amber tint
        "#eff6ff",  # Lane 4: Soft blue
        "#faf5ff",  # Lane 5: Soft purple
    ]
    
    lane_borders = [
        "#cbd5e1",
        "#bbf7d0",
        "#fde68a",
        "#bfdbfe",
        "#e9d5ff",
    ]

    # Process Colors
    blue_bg = "#eff6ff"
    blue_border = "#3b82f6"
    blue_text = "#1e40af"
    
    green_bg = "#ecfdf5"
    green_border = "#10b981"
    green_text = "#065f46"
    
    red_bg = "#fef2f2"
    red_border = "#ef4444"
    red_text = "#991b1b"
    
    amber_bg = "#fffbeb"
    amber_border = "#f59e0b"
    amber_text = "#92400e"
    
    gray_bg = "#f8fafc"
    gray_border = "#94a3b8"
    gray_text = "#1e293b"
    
    purple_bg = "#f5f3ff"
    purple_border = "#8b5cf6"
    purple_text = "#5b21b6"

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" style="background:{bg_canvas}; font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
  <defs>
    <!-- Drop Shadows for flowchart cards -->
    <filter id="shadow" x="-3%" y="-2%" width="106%" height="106%" filterUnits="userSpaceOnUse">
      <feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#0f172a" flood-opacity="0.08"/>
    </filter>
    <filter id="shadow-sm" x="-2%" y="-2%" width="104%" height="104%" filterUnits="userSpaceOnUse">
      <feDropShadow dx="0" dy="1" stdDeviation="2" flood-color="#0f172a" flood-opacity="0.06"/>
    </filter>
    <filter id="shadow-diamond" x="-10%" y="-10%" width="120%" height="120%" filterUnits="userSpaceOnUse">
      <feDropShadow dx="0" dy="2" stdDeviation="4" flood-color="#0284c7" flood-opacity="0.15"/>
    </filter>

    <!-- Arrow Markers -->
    <marker id="arr-dark" markerWidth="8" markerHeight="8" refX="5" refY="3" orient="auto">
      <path d="M0,0 L0,6 L6,3 z" fill="#334155"/>
    </marker>
    <marker id="arr-blue" markerWidth="8" markerHeight="8" refX="5" refY="3" orient="auto">
      <path d="M0,0 L0,6 L6,3 z" fill="#2563eb"/>
    </marker>
    <marker id="arr-green" markerWidth="8" markerHeight="8" refX="5" refY="3" orient="auto">
      <path d="M0,0 L0,6 L6,3 z" fill="#059669"/>
    </marker>
    <marker id="arr-red" markerWidth="8" markerHeight="8" refX="5" refY="3" orient="auto">
      <path d="M0,0 L0,6 L6,3 z" fill="#dc2626"/>
    </marker>
    <marker id="arr-amber" markerWidth="8" markerHeight="8" refX="5" refY="3" orient="auto">
      <path d="M0,0 L0,6 L6,3 z" fill="#d97706"/>
    </marker>
  </defs>

  <!-- ================= CANVAS BACKGROUND ================= -->
  <rect width="{width}" height="{height}" fill="{bg_canvas}"/>

  <!-- ================= TOP HEADER ================= -->
  <g id="flowchart-header" transform="translate(40, 20)">
    <text x="920" y="32" fill="{text_dark}" font-size="24" font-weight="800" text-anchor="middle" letter-spacing="-0.4">Post-Review Progress — HDFS Log Analysis &amp; DeepLog</text>
    <text x="920" y="52" fill="{text_muted}" font-size="12" font-weight="600" text-anchor="middle">TECHNICAL PROCESS FLOW: MENTOR FEEDBACK ➔ GROUND TRUTH ➔ EMPIRICAL SEQUENCES ➔ DEEPLOG ALIGNMENT ➔ ERROR ANALYSIS</text>
  </g>

  <!-- ================= GLOBAL PROCESS PROGRESSION CRUMB ================= -->
  <g id="breadcrumbs" transform="translate(40, 78)">
    <rect width="1840" height="26" rx="6" fill="#f1f5f9" stroke="#e2e8f0"/>
    <g transform="translate(18, 17)" font-size="9" font-weight="700">
      <text x="0" y="0" fill="#0f172a">START</text>
      <text x="44" y="0" fill="#94a3b8">➔</text>
      <text x="60" y="0" fill="#2563eb">MENTOR FEEDBACK</text>
      <text x="180" y="0" fill="#94a3b8">➔</text>
      <text x="196" y="0" fill="#059669">HDFS LOG STRUCTURE</text>
      <text x="338" y="0" fill="#94a3b8">➔</text>
      <text x="354" y="0" fill="#059669">GROUND TRUTH VALIDATION</text>
      <text x="526" y="0" fill="#94a3b8">➔</text>
      <text x="542" y="0" fill="#d97706">EVENT ANALYSIS</text>
      <text x="646" y="0" fill="#94a3b8">➔</text>
      <text x="662" y="0" fill="#d97706">SEQUENCE / TRANSITION ANALYSIS</text>
      <text x="872" y="0" fill="#94a3b8">➔</text>
      <text x="888" y="0" fill="#7c3aed">DEEPLOG ANALYSIS</text>
      <text x="1016" y="0" fill="#94a3b8">➔</text>
      <text x="1032" y="0" fill="#7c3aed">TOP-K DECISION</text>
      <text x="1144" y="0" fill="#94a3b8">➔</text>
      <text x="1160" y="0" fill="#2563eb">BLOCK-LEVEL EVALUATION</text>
      <text x="1324" y="0" fill="#94a3b8">➔</text>
      <text x="1340" y="0" fill="#ea580c">ERROR ANALYSIS</text>
      <text x="1454" y="0" fill="#94a3b8">➔</text>
      <text x="1470" y="0" fill="#ea580c">MODEL IMPROVEMENT</text>
      
      <!-- Right Badge -->
      <rect x="1680" y="-12" width="134" height="18" rx="4" fill="#1e293b"/>
      <text x="1747" y="0.5" fill="#ffffff" font-size="8.5" font-weight="700" text-anchor="middle">HORIZONTAL SWIMLANES</text>
    </g>
  </g>

  <!-- ================= 5 HORIZONTAL SWIMLANES =================
       Lane Width: 352px, Gap: 20px, Total Width: 1840px
       Start X: 40px
       Lane 1: x = 40
       Lane 2: x = 412
       Lane 3: x = 784
       Lane 4: x = 1156
       Lane 5: x = 1528
       Lane Top: y = 118, Lane Height: 934px (Ends at y = 1052)
  ============================================================= -->

  <!-- =========================================================
       LANE 1: Mentor Feedback (x: 40, w: 352)
  ========================================================== -->
  <g id="lane-1">
    <!-- Lane Background -->
    <rect x="40" y="118" width="352" height="934" rx="10" fill="{lane_bgs[0]}" stroke="{lane_borders[0]}" stroke-width="1.2"/>
    
    <!-- Lane Header -->
    <rect x="40" y="118" width="352" height="46" rx="10" fill="{lane_header_bg}"/>
    <rect x="40" y="154" width="352" height="10" fill="{lane_header_bg}"/>
    <rect x="40" y="118" width="5" height="46" rx="2.5" fill="#38bdf8"/>
    
    <text x="60" y="137" fill="#38bdf8" font-size="9" font-weight="800" letter-spacing="1">LANE 1</text>
    <text x="60" y="152" fill="{lane_header_text}" font-size="13" font-weight="800">Mentor Feedback</text>

    <!-- Node 1: START (Pill/Oval) -->
    <g transform="translate(146, 184)">
      <rect width="140" height="34" rx="17" fill="#0f172a" filter="url(#shadow)"/>
      <text x="70" y="22" fill="#ffffff" font-size="12" font-weight="800" text-anchor="middle" letter-spacing="1.2">START</text>
    </g>

    <!-- Arrow Start -> Mentor Review -->
    <line x1="216" y1="218" x2="216" y2="242" stroke="#334155" stroke-width="1.8" marker-end="url(#arr-dark)"/>

    <!-- Process: Mentor Review -->
    <g transform="translate(64, 244)">
      <rect width="304" height="60" rx="8" fill="{gray_bg}" stroke="{gray_border}" stroke-width="1.2" filter="url(#shadow)"/>
      <text x="152" y="26" fill="{text_dark}" font-size="12" font-weight="800" text-anchor="middle">Mentor Review</text>
      <text x="152" y="44" fill="{text_muted}" font-size="9.5" text-anchor="middle">Project evaluation &amp; system architecture audit</text>
    </g>

    <!-- Arrow Mentor Review -> Feedback Directive -->
    <line x1="216" y1="304" x2="216" y2="330" stroke="#334155" stroke-width="1.8" marker-end="url(#arr-dark)"/>

    <!-- Process: Feedback Directive (Amber) -->
    <g transform="translate(64, 332)">
      <rect width="304" height="106" rx="8" fill="{amber_bg}" stroke="{amber_border}" stroke-width="1.5" filter="url(#shadow)"/>
      
      <!-- Top banner inside -->
      <rect x="1" y="1" width="302" height="22" rx="7" fill="#fef3c7"/>
      <text x="152" y="16" fill="#b45309" font-size="9" font-weight="800" text-anchor="middle" letter-spacing="0.8">CRITICAL MENTOR DIRECTIVE</text>
      
      <text x="152" y="46" fill="{amber_text}" font-size="12" font-weight="800" text-anchor="middle">Feedback:</text>
      <text x="152" y="66" fill="#78350f" font-size="12.5" font-weight="800" text-anchor="middle">“Develop Deeper Understanding</text>
      <text x="152" y="84" fill="#78350f" font-size="12.5" font-weight="800" text-anchor="middle">of Logs”</text>
      <text x="152" y="100" fill="#92400e" font-size="8.5" text-anchor="middle">Gain true domain insight into HDFS behavior</text>
    </g>

    <!-- Arrow Feedback -> Define Focus -->
    <line x1="216" y1="438" x2="216" y2="466" stroke="#334155" stroke-width="1.8" marker-end="url(#arr-dark)"/>

    <!-- Process: Define Investigation Focus -->
    <g transform="translate(64, 468)">
      <rect width="304" height="194" rx="8" fill="{blue_bg}" stroke="{blue_border}" stroke-width="1.2" filter="url(#shadow)"/>
      <rect x="1" y="1" width="302" height="24" rx="7" fill="#dbeafe"/>
      <text x="152" y="17" fill="{blue_text}" font-size="10.5" font-weight="800" text-anchor="middle">Define Investigation Focus</text>
      
      <!-- 4 Pillars -->
      <g transform="translate(16, 36)">
        <!-- 1 -->
        <g transform="translate(0, 0)">
          <rect width="272" height="32" rx="5" fill="#ffffff" stroke="#bfdbfe"/>
          <text x="12" y="15" fill="{text_dark}" font-size="9.5" font-weight="800">• Event Meaning</text>
          <text x="12" y="26" fill="{text_muted}" font-size="8">What each log template represents in HDFS</text>
        </g>
        <!-- 2 -->
        <g transform="translate(0, 38)">
          <rect width="272" height="32" rx="5" fill="#ffffff" stroke="#bfdbfe"/>
          <text x="12" y="15" fill="{text_dark}" font-size="9.5" font-weight="800">• Ground Truth</text>
          <text x="12" y="26" fill="{text_muted}" font-size="8">Cross-check trace labels &amp; verify integrity</text>
        </g>
        <!-- 3 -->
        <g transform="translate(0, 76)">
          <rect width="272" height="32" rx="5" fill="#ffffff" stroke="#bfdbfe"/>
          <text x="12" y="15" fill="{text_dark}" font-size="9.5" font-weight="800">• Sequence Behavior</text>
          <text x="12" y="26" fill="{text_muted}" font-size="8">Study execution ordering &amp; 2-gram transitions</text>
        </g>
        <!-- 4 -->
        <g transform="translate(0, 114)">
          <rect width="272" height="32" rx="5" fill="#ffffff" stroke="#bfdbfe"/>
          <text x="12" y="15" fill="{text_dark}" font-size="9.5" font-weight="800">• Model Errors</text>
          <text x="12" y="26" fill="{text_muted}" font-size="8">Analyze False Positives &amp; False Negatives</text>
        </g>
      </g>
    </g>

    <!-- Lane 1 Status Note at Bottom -->
    <g transform="translate(64, 980)">
      <rect width="304" height="48" rx="6" fill="#f1f5f9" stroke="#cbd5e1"/>
      <text x="152" y="20" fill="{text_muted}" font-size="8.5" font-weight="700" text-anchor="middle">MENTOR DIRECTIVE PHASE COMPLETE</text>
      <text x="152" y="36" fill="{text_dark}" font-size="9" font-weight="600" text-anchor="middle">Directs analytical methodology into Lane 2</text>
    </g>
  </g>

  <!-- ================= ELBOW ARROW: LANE 1 -> LANE 2 ================= -->
  <!-- From Define Focus (right edge: x=368, y=565) over to Lane 2 HDFS Dataset (left edge: x=436, y=214) -->
  <path d="M 368 565 L 390 565 L 390 214 L 434 214" fill="none" stroke="#2563eb" stroke-width="2" marker-end="url(#arr-blue)"/>
  <rect x="380" y="375" width="22" height="18" rx="3" fill="#2563eb"/>
  <text x="391" y="388" fill="#ffffff" font-size="9" font-weight="800" text-anchor="middle">➔</text>

  <!-- =========================================================
       LANE 2: HDFS Dataset & Log Structure (x: 412, w: 352)
  ========================================================== -->
  <g id="lane-2">
    <!-- Lane Background -->
    <rect x="412" y="118" width="352" height="934" rx="10" fill="{lane_bgs[1]}" stroke="{lane_borders[1]}" stroke-width="1.2"/>
    
    <!-- Lane Header -->
    <rect x="412" y="118" width="352" height="46" rx="10" fill="{lane_header_bg}"/>
    <rect x="412" y="154" width="352" height="10" fill="{lane_header_bg}"/>
    <rect x="412" y="118" width="5" height="46" rx="2.5" fill="#34d399"/>
    
    <text x="432" y="137" fill="#34d399" font-size="9" font-weight="800" letter-spacing="1">LANE 2</text>
    <text x="432" y="152" fill="{lane_header_text}" font-size="13" font-weight="800">HDFS Dataset &amp; Log Structure</text>

    <!-- Process: HDFS_v1 Dataset -->
    <g transform="translate(436, 184)">
      <rect width="304" height="60" rx="8" fill="{gray_bg}" stroke="{gray_border}" stroke-width="1.2" filter="url(#shadow)"/>
      <text x="152" y="26" fill="{text_dark}" font-size="12" font-weight="800" text-anchor="middle">HDFS_v1 Dataset</text>
      <text x="152" y="44" fill="{text_muted}" font-size="9" text-anchor="middle">11,175,629 raw lines • 575,061 block sessions</text>
    </g>

    <!-- Arrow HDFS_v1 -> Log Template -->
    <line x1="588" y1="244" x2="588" y2="268" stroke="#334155" stroke-width="1.8" marker-end="url(#arr-dark)"/>

    <!-- Process: Log Template -> Event ID -->
    <g transform="translate(436, 270)">
      <rect width="304" height="178" rx="8" fill="#ffffff" stroke="{border_subtle}" stroke-width="1.2" filter="url(#shadow)"/>
      
      <rect x="1" y="1" width="302" height="24" rx="7" fill="#f1f5f9"/>
      <text x="152" y="17" fill="{text_dark}" font-size="10.5" font-weight="800" text-anchor="middle">Log Template ➔ Event ID</text>
      
      <!-- Examples (2 cols x 3 rows) -->
      <g transform="translate(12, 34)">
        <!-- Col 1 -->
        <g transform="translate(0, 0)">
          <rect width="136" height="38" rx="4" fill="#f8fafc" stroke="#e2e8f0"/>
          <text x="8" y="16" fill="{blue_text}" font-size="9.5" font-weight="800">E5 =</text>
          <text x="32" y="16" fill="{text_dark}" font-size="9" font-weight="600">Receiving block</text>
          <text x="8" y="29" fill="{text_muted}" font-size="7.5">Inbound stream</text>
        </g>
        <g transform="translate(0, 44)">
          <rect width="136" height="38" rx="4" fill="#f8fafc" stroke="#e2e8f0"/>
          <text x="8" y="16" fill="{blue_text}" font-size="9.5" font-weight="800">E9 =</text>
          <text x="32" y="16" fill="{text_dark}" font-size="9" font-weight="600">Received block</text>
          <text x="8" y="29" fill="{text_muted}" font-size="7.5">Block stored ack</text>
        </g>
        <g transform="translate(0, 88)">
          <rect width="136" height="46" rx="4" fill="#f8fafc" stroke="#e2e8f0"/>
          <text x="8" y="16" fill="{blue_text}" font-size="9.5" font-weight="800">E11 =</text>
          <text x="34" y="16" fill="{text_dark}" font-size="8.5" font-weight="600">PacketResponder</text>
          <text x="8" y="29" fill="{text_dark}" font-size="8.5" font-weight="600">terminating</text>
          <text x="8" y="40" fill="{text_muted}" font-size="7.5">Stream finished</text>
        </g>

        <!-- Col 2 -->
        <g transform="translate(144, 0)">
          <rect width="136" height="38" rx="4" fill="#f8fafc" stroke="#e2e8f0"/>
          <text x="8" y="16" fill="{blue_text}" font-size="9.5" font-weight="800">E21 =</text>
          <text x="38" y="16" fill="{text_dark}" font-size="8.5" font-weight="600">Deleting block</text>
          <text x="8" y="29" fill="{text_muted}" font-size="7.5">Storage file clean</text>
        </g>
        <g transform="translate(144, 44)">
          <rect width="136" height="38" rx="4" fill="#f8fafc" stroke="#e2e8f0"/>
          <text x="8" y="16" fill="{blue_text}" font-size="9.5" font-weight="800">E22 =</text>
          <text x="38" y="16" fill="{text_dark}" font-size="8.5" font-weight="600">allocateBlock</text>
          <text x="8" y="29" fill="{text_muted}" font-size="7.5">NameNode alloc</text>
        </g>
        <g transform="translate(144, 88)">
          <rect width="136" height="46" rx="4" fill="#f8fafc" stroke="#e2e8f0"/>
          <text x="8" y="16" fill="{blue_text}" font-size="9.5" font-weight="800">E26 =</text>
          <text x="38" y="16" fill="{text_dark}" font-size="8.5" font-weight="600">addStored-</text>
          <text x="8" y="29" fill="{text_dark}" font-size="8.5" font-weight="600">Block</text>
          <text x="8" y="40" fill="{text_muted}" font-size="7.5">Replica verified</text>
        </g>
      </g>
    </g>

    <!-- Arrow Template -> Sequence -->
    <line x1="588" y1="448" x2="588" y2="472" stroke="#334155" stroke-width="1.8" marker-end="url(#arr-dark)"/>

    <!-- Process: Block ID -> Event Sequence -->
    <g transform="translate(436, 474)">
      <rect width="304" height="84" rx="8" fill="#ffffff" stroke="{border_subtle}" stroke-width="1.2" filter="url(#shadow)"/>
      <text x="152" y="22" fill="{text_dark}" font-size="10.5" font-weight="800" text-anchor="middle">Block ID ➔ Event Sequence</text>
      
      <!-- Sequence Example -->
      <g transform="translate(10, 32)">
        <rect width="284" height="28" rx="4" fill="#f1f5f9" stroke="#cbd5e1"/>
        <text x="142" y="18" fill="{blue_text}" font-size="8.5" font-weight="800" text-anchor="middle">E5 ➔ E22 ➔ E11 ➔ E9 ➔ E26 ➔ E23 ➔ E21</text>
      </g>
      <text x="152" y="74" fill="{text_muted}" font-size="8" text-anchor="middle">Ordered temporal execution trail per block</text>
    </g>

    <!-- Arrow Sequence -> Ground Truth -->
    <line x1="588" y1="558" x2="588" y2="582" stroke="#334155" stroke-width="1.8" marker-end="url(#arr-dark)"/>

    <!-- Process: Ground Truth -->
    <g transform="translate(436, 584)">
      <rect width="304" height="74" rx="8" fill="#ffffff" stroke="{border_subtle}" stroke-width="1.2" filter="url(#shadow)"/>
      <text x="152" y="20" fill="{text_dark}" font-size="10.5" font-weight="800" text-anchor="middle">Ground Truth</text>
      
      <!-- Split -->
      <g transform="translate(12, 30)">
        <rect width="134" height="34" rx="4" fill="{blue_bg}" stroke="{blue_border}"/>
        <text x="67" y="15" fill="{blue_text}" font-size="8" font-weight="700" text-anchor="middle">NORMAL BLOCKS</text>
        <text x="67" y="28" fill="{blue_text}" font-size="11" font-weight="800" text-anchor="middle">558,223</text>

        <rect x="146" y="0" width="134" height="34" rx="4" fill="{red_bg}" stroke="{red_border}"/>
        <text x="213" y="15" fill="{red_text}" font-size="8" font-weight="700" text-anchor="middle">ANOMALY BLOCKS</text>
        <text x="213" y="28" fill="{red_text}" font-size="11" font-weight="800" text-anchor="middle">16,838</text>
      </g>
    </g>

    <!-- Arrow Ground Truth -> Decision Diamond -->
    <line x1="588" y1="658" x2="588" y2="682" stroke="#334155" stroke-width="1.8" marker-end="url(#arr-dark)"/>

    <!-- DECISION DIAMOND: Does Event Trace Match Ground Truth? -->
    <g transform="translate(588, 730)">
      <!-- Diamond Shape: centered at (0, 0), width 190, height 76 -->
      <polygon points="0,-38 95,0 0,38 -95,0" fill="#f0fdf4" stroke="#10b981" stroke-width="1.8" filter="url(#shadow-diamond)"/>
      <text x="0" y="-8" fill="{text_dark}" font-size="9" font-weight="800" text-anchor="middle">Does Event Trace Match</text>
      <text x="0" y="6" fill="{text_dark}" font-size="9" font-weight="800" text-anchor="middle">Ground Truth?</text>
      <text x="0" y="20" fill="#059669" font-size="8" font-weight="800" text-anchor="middle">(Cross-Validation)</text>
    </g>

    <!-- Branch YES (Down Arrow) -->
    <line x1="588" y1="768" x2="588" y2="800" stroke="#059669" stroke-width="2" marker-end="url(#arr-green)"/>
    <rect x="596" y="776" width="30" height="16" rx="3" fill="#ecfdf5" stroke="#10b981"/>
    <text x="611" y="788" fill="#047857" font-size="8.5" font-weight="800" text-anchor="middle">YES</text>

    <!-- Process: Validation Match Result -->
    <g transform="translate(436, 802)">
      <rect width="304" height="120" rx="8" fill="{green_bg}" stroke="{green_border}" stroke-width="1.5" filter="url(#shadow)"/>
      
      <rect x="1" y="1" width="302" height="22" rx="7" fill="#d1fae5"/>
      <text x="152" y="16" fill="{green_text}" font-size="9" font-weight="800" text-anchor="middle" letter-spacing="0.6">100% LABEL AUDIT VALIDATED</text>
      
      <g transform="translate(12, 32)">
        <rect width="134" height="24" rx="4" fill="#ffffff" stroke="#a7f3d0"/>
        <text x="67" y="16" fill="{green_text}" font-size="8.5" font-weight="700" text-anchor="middle">Normal ↔ Success</text>

        <rect x="146" y="0" width="134" height="24" rx="4" fill="#ffffff" stroke="#fecaca"/>
        <text x="213" y="16" fill="{red_text}" font-size="8.5" font-weight="700" text-anchor="middle">Anomaly ↔ Fail</text>
      </g>

      <g transform="translate(0, 68)">
        <text x="152" y="16" fill="{green_text}" font-size="11.5" font-weight="800" text-anchor="middle">“575,061 blocks cross-checked”</text>
        <text x="152" y="32" fill="#065f46" font-size="11.5" font-weight="800" text-anchor="middle">“No label mismatch”</text>
      </g>
    </g>

    <!-- Lane 2 Bottom Note -->
    <g transform="translate(436, 980)">
      <rect width="304" height="48" rx="6" fill="#f1f5f9" stroke="#cbd5e1"/>
      <text x="152" y="20" fill="{green_text}" font-size="8.5" font-weight="700" text-anchor="middle">GROUND TRUTH RIGOROUSLY ESTABLISHED</text>
      <text x="152" y="36" fill="{text_dark}" font-size="9" font-weight="600" text-anchor="middle">Enables valid frequency &amp; sequence analysis</text>
    </g>
  </g>

  <!-- ================= ELBOW ARROW: LANE 2 -> LANE 3 ================= -->
  <!-- From Lane 2 Validation (right edge: x=740, y=862) to Lane 3 Frequency (left edge: x=808, y=214) -->
  <path d="M 740 862 L 762 862 L 762 214 L 806 214" fill="none" stroke="#059669" stroke-width="2" marker-end="url(#arr-green)"/>
  <rect x="752" y="530" width="20" height="18" rx="3" fill="#059669"/>
  <text x="762" y="543" fill="#ffffff" font-size="9" font-weight="800" text-anchor="middle">➔</text>

  <!-- =========================================================
       LANE 3: Event & Sequence Analysis (x: 784, w: 352)
  ========================================================== -->
  <g id="lane-3">
    <!-- Lane Background -->
    <rect x="784" y="118" width="352" height="934" rx="10" fill="{lane_bgs[2]}" stroke="{lane_borders[2]}" stroke-width="1.2"/>
    
    <!-- Lane Header -->
    <rect x="784" y="118" width="352" height="46" rx="10" fill="{lane_header_bg}"/>
    <rect x="784" y="154" width="352" height="10" fill="{lane_header_bg}"/>
    <rect x="784" y="118" width="5" height="46" rx="2.5" fill="#fbbf24"/>
    
    <text x="804" y="137" fill="#fbbf24" font-size="9" font-weight="800" letter-spacing="1">LANE 3</text>
    <text x="804" y="152" fill="{lane_header_text}" font-size="13" font-weight="800">Event &amp; Sequence Analysis</text>

    <!-- Process: Event Frequency Analysis -->
    <g transform="translate(808, 184)">
      <rect width="304" height="60" rx="8" fill="{amber_bg}" stroke="{amber_border}" stroke-width="1.2" filter="url(#shadow)"/>
      <text x="152" y="26" fill="{text_dark}" font-size="12" font-weight="800" text-anchor="middle">Event Frequency Analysis</text>
      <text x="152" y="44" fill="{text_muted}" font-size="9" text-anchor="middle">Normal vs Anomaly count distribution</text>
    </g>

    <!-- Arrow Frequency -> Presence -->
    <line x1="960" y1="244" x2="960" y2="268" stroke="#334155" stroke-width="1.8" marker-end="url(#arr-dark)"/>

    <!-- Process: Event Presence Analysis (Highlights E20 and E7) -->
    <g transform="translate(808, 270)">
      <rect width="304" height="210" rx="8" fill="#ffffff" stroke="{border_subtle}" stroke-width="1.2" filter="url(#shadow)"/>
      
      <rect x="1" y="1" width="302" height="24" rx="7" fill="#fef3c7"/>
      <text x="152" y="17" fill="{amber_text}" font-size="10.5" font-weight="800" text-anchor="middle">Event Presence Analysis</text>
      
      <text x="152" y="38" fill="{text_muted}" font-size="8.5" text-anchor="middle">Metric: % of blocks containing event at least once</text>

      <!-- Highlight finding E20 -->
      <g transform="translate(12, 48)">
        <rect width="280" height="64" rx="6" fill="{red_bg}" stroke="{red_border}" stroke-width="1.2"/>
        <rect x="8" y="8" width="38" height="20" rx="3" fill="{red_text}"/>
        <text x="27" y="22" fill="#ffffff" font-size="10" font-weight="800" text-anchor="middle">E20</text>
        <text x="54" y="22" fill="{red_text}" font-size="10" font-weight="800">Significant Presence Skew</text>

        <!-- Stats row -->
        <g transform="translate(8, 34)">
          <rect width="128" height="22" rx="3" fill="#ffffff" stroke="#cbd5e1"/>
          <text x="64" y="15" fill="{blue_text}" font-size="9" font-weight="700" text-anchor="middle">Normal = 0.04%</text>

          <rect x="136" y="0" width="128" height="22" rx="3" fill="#ffffff" stroke="{red_border}"/>
          <text x="200" y="15" fill="{red_text}" font-size="9" font-weight="800" text-anchor="middle">Anomaly = 30.44%</text>
        </g>
      </g>

      <!-- Highlight finding E7 -->
      <g transform="translate(12, 120)">
        <rect width="280" height="64" rx="6" fill="{red_bg}" stroke="{red_border}" stroke-width="1.2"/>
        <rect x="8" y="8" width="38" height="20" rx="3" fill="{red_text}"/>
        <text x="27" y="22" fill="#ffffff" font-size="10" font-weight="800" text-anchor="middle">E7</text>
        <text x="54" y="22" fill="{red_text}" font-size="10" font-weight="800">Zero Normal Occurrence</text>

        <!-- Stats row -->
        <g transform="translate(8, 34)">
          <rect width="128" height="22" rx="3" fill="#ffffff" stroke="#cbd5e1"/>
          <text x="64" y="15" fill="{blue_text}" font-size="9" font-weight="700" text-anchor="middle">Normal = 0.00%</text>

          <rect x="136" y="0" width="128" height="22" rx="3" fill="#ffffff" stroke="{red_border}"/>
          <text x="200" y="15" fill="{red_text}" font-size="9" font-weight="800" text-anchor="middle">Anomaly = 19.62%</text>
        </g>
      </g>

      <text x="152" y="200" fill="{text_muted}" font-size="8" text-anchor="middle">Shows events strongly associated with anomaly blocks</text>
    </g>

    <!-- Arrow Presence -> Transition -->
    <line x1="960" y1="480" x2="960" y2="504" stroke="#334155" stroke-width="1.8" marker-end="url(#arr-dark)"/>

    <!-- Process: Event Transition Analysis -->
    <g transform="translate(808, 506)">
      <rect width="304" height="150" rx="8" fill="#ffffff" stroke="{border_subtle}" stroke-width="1.2" filter="url(#shadow)"/>
      
      <rect x="1" y="1" width="302" height="24" rx="7" fill="#f8fafc"/>
      <text x="152" y="17" fill="{text_dark}" font-size="10.5" font-weight="800" text-anchor="middle">Event Transition Analysis</text>
      
      <text x="152" y="38" fill="{text_muted}" font-size="8.5" text-anchor="middle">Sequential pairs: Event A ➔ Event B</text>

      <!-- 3 Normal Transition Examples -->
      <g transform="translate(12, 48)">
        <g transform="translate(0, 0)">
          <rect width="280" height="26" rx="4" fill="#f1f5f9" stroke="#cbd5e1"/>
          <text x="12" y="17" fill="{blue_text}" font-size="9" font-weight="800">E5 ➔ E22</text>
          <text x="268" y="16.5" fill="{text_muted}" font-size="8" text-anchor="end">Normal allocate flow</text>
        </g>
        <g transform="translate(0, 32)">
          <rect width="280" height="26" rx="4" fill="#f1f5f9" stroke="#cbd5e1"/>
          <text x="12" y="17" fill="{blue_text}" font-size="9" font-weight="800">E11 ➔ E9</text>
          <text x="268" y="16.5" fill="{text_muted}" font-size="8" text-anchor="end">PacketResponder terminate</text>
        </g>
        <g transform="translate(0, 64)">
          <rect width="280" height="26" rx="4" fill="#f1f5f9" stroke="#cbd5e1"/>
          <text x="12" y="17" fill="{blue_text}" font-size="9" font-weight="800">E9 ➔ E11</text>
          <text x="268" y="16.5" fill="{text_muted}" font-size="8" text-anchor="end">Packet stream coordination</text>
        </g>
      </g>
    </g>

    <!-- Arrow Transition -> Key Finding E5 -> E7 -->
    <line x1="960" y1="656" x2="960" y2="680" stroke="#334155" stroke-width="1.8" marker-end="url(#arr-dark)"/>

    <!-- Process: Highlight E5 -> E7 -->
    <g transform="translate(808, 682)">
      <rect width="304" height="154" rx="8" fill="{red_bg}" stroke="{red_border}" stroke-width="1.5" filter="url(#shadow)"/>
      
      <rect x="1" y="1" width="302" height="22" rx="7" fill="#fee2e2"/>
      <text x="152" y="16" fill="{red_text}" font-size="9" font-weight="800" text-anchor="middle" letter-spacing="0.6">KEY ANOMALY TRANSITION FINDING</text>
      
      <text x="152" y="44" fill="{red_text}" font-size="14" font-weight="800" text-anchor="middle">“E5 ➔ E7”</text>

      <!-- Stats Box -->
      <g transform="translate(12, 54)">
        <rect width="134" height="36" rx="4" fill="#ffffff" stroke="#cbd5e1"/>
        <text x="67" y="15" fill="#64748b" font-size="8" font-weight="700" text-anchor="middle">NORMAL BLOCKS</text>
        <text x="67" y="30" fill="{blue_text}" font-size="11.5" font-weight="800" text-anchor="middle">Normal = 0</text>

        <rect x="146" y="0" width="134" height="36" rx="4" fill="#ffffff" stroke="{red_border}"/>
        <text x="213" y="15" fill="#991b1b" font-size="8" font-weight="700" text-anchor="middle">ANOMALY BLOCKS</text>
        <text x="213" y="30" fill="{red_text}" font-size="11.5" font-weight="800" text-anchor="middle">Anomaly = 1,993</text>
      </g>

      <text x="152" y="109" fill="{red_text}" font-size="9.5" font-weight="800" text-anchor="middle">“Anomaly-associated transition pattern”</text>
      
      <!-- Scientific Rigor Note -->
      <rect x="12" y="120" width="280" height="24" rx="4" fill="#fffbeb" stroke="#fcd34d"/>
      <text x="152" y="136" fill="#92400e" font-size="8" font-weight="700" text-anchor="middle">Association, not proof of causation.</text>
    </g>

    <!-- Lane 3 Bottom Note -->
    <g transform="translate(808, 980)">
      <rect width="304" height="48" rx="6" fill="#f1f5f9" stroke="#cbd5e1"/>
      <text x="152" y="20" fill="{amber_text}" font-size="8.5" font-weight="700" text-anchor="middle">SEQUENTIAL TRANSITIONS CONFIRMED</text>
      <text x="152" y="36" fill="{text_dark}" font-size="9" font-weight="600" text-anchor="middle">Transitions inform DeepLog sequential modeling</text>
    </g>
  </g>

  <!-- ================= ELBOW ARROW: LANE 3 -> LANE 4 ================= -->
  <!-- From Lane 3 Transition Finding (right edge: x=1112, y=759) to Lane 4 Canonical Mapping (left edge: x=1180, y=214) -->
  <path d="M 1112 759 L 1134 759 L 1134 214 L 1178 214" fill="none" stroke="#d97706" stroke-width="2" marker-end="url(#arr-amber)"/>
  <rect x="1124" y="480" width="20" height="18" rx="3" fill="#d97706"/>
  <text x="1134" y="493" fill="#ffffff" font-size="9" font-weight="800" text-anchor="middle">➔</text>

  <!-- =========================================================
       LANE 4: DeepLog Model Analysis (x: 1156, w: 352)
  ========================================================== -->
  <g id="lane-4">
    <!-- Lane Background -->
    <rect x="1156" y="118" width="352" height="934" rx="10" fill="{lane_bgs[3]}" stroke="{lane_borders[3]}" stroke-width="1.2"/>
    
    <!-- Lane Header -->
    <rect x="1156" y="118" width="352" height="46" rx="10" fill="{lane_header_bg}"/>
    <rect x="1156" y="154" width="352" height="10" fill="{lane_header_bg}"/>
    <rect x="1156" y="118" width="5" height="46" rx="2.5" fill="#818cf8"/>
    
    <text x="1176" y="137" fill="#818cf8" font-size="9" font-weight="800" letter-spacing="1">LANE 4</text>
    <text x="1176" y="152" fill="{lane_header_text}" font-size="13" font-weight="800">DeepLog Model Analysis</text>

    <!-- Process: Canonical Training Mapping -->
    <g transform="translate(1180, 184)">
      <rect width="304" height="60" rx="8" fill="{blue_bg}" stroke="{blue_border}" stroke-width="1.2" filter="url(#shadow)"/>
      <text x="152" y="24" fill="{blue_text}" font-size="11" font-weight="800" text-anchor="middle">Canonical Training Mapping</text>
      <text x="152" y="42" fill="{text_dark}" font-size="9.5" font-weight="600" text-anchor="middle">HDFS Event ID ➔ DeepLog Numeric ID</text>
    </g>

    <!-- Arrow Mapping -> Context Window -->
    <line x1="1332" y1="244" x2="1332" y2="268" stroke="#334155" stroke-width="1.8" marker-end="url(#arr-dark)"/>

    <!-- Process: Context Window = 10 Events -->
    <g transform="translate(1180, 270)">
      <rect width="304" height="52" rx="8" fill="{gray_bg}" stroke="{gray_border}" stroke-width="1.2" filter="url(#shadow)"/>
      <text x="152" y="24" fill="{text_dark}" font-size="11.5" font-weight="800" text-anchor="middle">Context Window = 10 Events</text>
      <text x="152" y="40" fill="{text_muted}" font-size="8.5" text-anchor="middle">10-step history sliding window for next-token forecast</text>
    </g>

    <!-- Arrow Window -> LSTM -->
    <line x1="1332" y1="322" x2="1332" y2="346" stroke="#334155" stroke-width="1.8" marker-end="url(#arr-dark)"/>

    <!-- Process: DeepLog 2-Layer LSTM -->
    <g transform="translate(1180, 348)">
      <rect width="304" height="78" rx="8" fill="{purple_bg}" stroke="{purple_border}" stroke-width="1.5" filter="url(#shadow)"/>
      <rect x="1" y="1" width="302" height="20" rx="7" fill="#ede9fe"/>
      <text x="152" y="15" fill="{purple_text}" font-size="9" font-weight="800" text-anchor="middle" letter-spacing="0.6">DEEPLOG CORE NEURAL MODEL</text>
      
      <text x="152" y="40" fill="{purple_text}" font-size="12" font-weight="800" text-anchor="middle">DeepLog</text>
      <text x="152" y="56" fill="{text_dark}" font-size="10.5" font-weight="700" text-anchor="middle">2-Layer LSTM</text>
      <text x="152" y="70" fill="{text_muted}" font-size="9" text-anchor="middle">Hidden Size = 128</text>
    </g>

    <!-- Arrow LSTM -> Next-Event Prediction -->
    <line x1="1332" y1="426" x2="1332" y2="450" stroke="#334155" stroke-width="1.8" marker-end="url(#arr-dark)"/>

    <!-- Process: Next-Event Prediction -->
    <g transform="translate(1180, 452)">
      <rect width="304" height="52" rx="8" fill="{gray_bg}" stroke="{gray_border}" stroke-width="1.2" filter="url(#shadow)"/>
      <text x="152" y="24" fill="{text_dark}" font-size="11" font-weight="800" text-anchor="middle">Next-Event Prediction</text>
      <text x="152" y="40" fill="{text_muted}" font-size="8.5" text-anchor="middle">Softmax probability across event vocabulary</text>
    </g>

    <!-- Arrow Next-Event -> Top-K Prediction -->
    <line x1="1332" y1="504" x2="1332" y2="528" stroke="#334155" stroke-width="1.8" marker-end="url(#arr-dark)"/>

    <!-- Process: Top-K Prediction -->
    <g transform="translate(1180, 530)">
      <rect width="304" height="52" rx="8" fill="{blue_bg}" stroke="{blue_border}" stroke-width="1.2" filter="url(#shadow)"/>
      <text x="152" y="24" fill="{blue_text}" font-size="11" font-weight="800" text-anchor="middle">Top-K Prediction</text>
      <text x="152" y="40" fill="{text_muted}" font-size="8.5" text-anchor="middle">Candidate pool of K most probable events</text>
    </g>

    <!-- Arrow Top-K -> Decision Diamond -->
    <line x1="1332" y1="582" x2="1332" y2="608" stroke="#334155" stroke-width="1.8" marker-end="url(#arr-dark)"/>

    <!-- DECISION DIAMOND: Actual Event in Top-K? -->
    <g transform="translate(1332, 650)">
      <polygon points="0,-36 90,0 0,36 -90,0" fill="#eff6ff" stroke="#3b82f6" stroke-width="1.8" filter="url(#shadow-diamond)"/>
      <text x="0" y="-8" fill="{text_dark}" font-size="9.5" font-weight="800" text-anchor="middle">Actual Event in</text>
      <text x="0" y="8" fill="{text_dark}" font-size="9.5" font-weight="800" text-anchor="middle">Top-K?</text>
      <text x="0" y="20" fill="{blue_text}" font-size="7.5" font-weight="700" text-anchor="middle">(Anomaly Test)</text>
    </g>

    <!-- Branch YES (Left/Green) -->
    <path d="M 1242 650 L 1206 650 L 1206 708 L 1238 708" fill="none" stroke="#059669" stroke-width="1.8" marker-end="url(#arr-green)"/>
    <rect x="1192" y="664" width="28" height="16" rx="3" fill="#ecfdf5" stroke="#10b981"/>
    <text x="1206" y="676" fill="#047857" font-size="8.5" font-weight="800" text-anchor="middle">YES</text>

    <!-- Box YES: Normal Event -->
    <g transform="translate(1244, 694)">
      <rect width="100" height="28" rx="4" fill="{green_bg}" stroke="{green_border}"/>
      <text x="50" y="18" fill="{green_text}" font-size="9" font-weight="800" text-anchor="middle">Normal Event</text>
    </g>

    <!-- Branch NO (Down/Right/Red) -->
    <path d="M 1332 686 L 1332 730" fill="none" stroke="#dc2626" stroke-width="2" marker-end="url(#arr-red)"/>
    <rect x="1342" y="696" width="26" height="16" rx="3" fill="#fef2f2" stroke="#ef4444"/>
    <text x="1355" y="708" fill="#991b1b" font-size="8.5" font-weight="800" text-anchor="middle">NO</text>

    <!-- Box NO: Event Anomaly -->
    <g transform="translate(1282, 732)">
      <rect width="100" height="28" rx="4" fill="{red_bg}" stroke="{red_border}"/>
      <text x="50" y="18" fill="{red_text}" font-size="9" font-weight="800" text-anchor="middle">Event Anomaly</text>
    </g>

    <!-- Arrow Event Anomaly -> Block Anomaly -->
    <line x1="1332" y1="760" x2="1332" y2="784" stroke="#dc2626" stroke-width="1.8" marker-end="url(#arr-red)"/>

    <!-- Process: Event Anomaly -> Block Anomaly -->
    <g transform="translate(1180, 786)">
      <rect width="304" height="66" rx="8" fill="{red_bg}" stroke="{red_border}" stroke-width="1.5" filter="url(#shadow)"/>
      <text x="152" y="24" fill="{red_text}" font-size="11.5" font-weight="800" text-anchor="middle">Event Anomaly ➔ Block Anomaly</text>
      <text x="152" y="42" fill="{text_dark}" font-size="9" font-weight="600" text-anchor="middle">Any unexpected event in sequence flags</text>
      <text x="152" y="56" fill="{text_dark}" font-size="9" font-weight="600" text-anchor="middle">the entire block as ANOMALY</text>
    </g>

    <!-- Lane 4 Bottom Note -->
    <g transform="translate(1180, 980)">
      <rect width="304" height="48" rx="6" fill="#f1f5f9" stroke="#cbd5e1"/>
      <text x="152" y="20" fill="{purple_text}" font-size="8.5" font-weight="700" text-anchor="middle">DEEPLOG INFERENCE MECHANISM</text>
      <text x="152" y="36" fill="{text_dark}" font-size="9" font-weight="600" text-anchor="middle">Proceeds to empirical block evaluation</text>
    </g>
  </g>

  <!-- ================= ELBOW ARROW: LANE 4 -> LANE 5 ================= -->
  <!-- From Lane 4 Block Anomaly (right edge: x=1484, y=819) to Lane 5 Block Evaluation (left edge: x=1552, y=214) -->
  <path d="M 1484 819 L 1506 819 L 1506 214 L 1550 214" fill="none" stroke="#dc2626" stroke-width="2" marker-end="url(#arr-red)"/>
  <rect x="1496" y="516" width="20" height="18" rx="3" fill="#dc2626"/>
  <text x="1506" y="529" fill="#ffffff" font-size="9" font-weight="800" text-anchor="middle">➔</text>

  <!-- =========================================================
       LANE 5: Evaluation & Next Step (x: 1528, w: 352)
  ========================================================== -->
  <g id="lane-5">
    <!-- Lane Background -->
    <rect x="1528" y="118" width="352" height="934" rx="10" fill="{lane_bgs[4]}" stroke="{lane_borders[4]}" stroke-width="1.2"/>
    
    <!-- Lane Header -->
    <rect x="1528" y="118" width="352" height="46" rx="10" fill="{lane_header_bg}"/>
    <rect x="1528" y="154" width="352" height="10" fill="{lane_header_bg}"/>
    <rect x="1528" y="118" width="5" height="46" rx="2.5" fill="#f472b6"/>
    
    <text x="1548" y="137" fill="#f472b6" font-size="9" font-weight="800" letter-spacing="1">LANE 5</text>
    <text x="1548" y="152" fill="{lane_header_text}" font-size="13" font-weight="800">Evaluation &amp; Next Step</text>

    <!-- Process: Block-Level Evaluation Top-5 -->
    <g transform="translate(1552, 184)">
      <rect width="304" height="142" rx="8" fill="#ffffff" stroke="{border_subtle}" stroke-width="1.2" filter="url(#shadow)"/>
      
      <rect x="1" y="1" width="302" height="22" rx="7" fill="#fdf2f8"/>
      <text x="152" y="16" fill="#be185d" font-size="9" font-weight="800" text-anchor="middle" letter-spacing="0.6">BLOCK-LEVEL EVALUATION (TOP-5)</text>
      
      <!-- Compact Results Grid -->
      <g transform="translate(12, 30)">
        <!-- Row 1: Accuracy & Precision -->
        <g transform="translate(0, 0)">
          <rect width="136" height="26" rx="4" fill="{green_bg}" stroke="{green_border}"/>
          <text x="68" y="17" fill="{green_text}" font-size="9" font-weight="800" text-anchor="middle">Accuracy = 98.53%</text>

          <rect x="144" y="0" width="136" height="26" rx="4" fill="#f8fafc" stroke="#cbd5e1"/>
          <text x="212" y="17" fill="{text_dark}" font-size="9" font-weight="700" text-anchor="middle">Precision = 85.73%</text>
        </g>
        <!-- Row 2: Recall & F1 -->
        <g transform="translate(0, 32)">
          <rect width="136" height="26" rx="4" fill="#f8fafc" stroke="#cbd5e1"/>
          <text x="68" y="17" fill="{text_dark}" font-size="9" font-weight="700" text-anchor="middle">Recall = 60.07%</text>

          <rect x="144" y="0" width="136" height="26" rx="4" fill="{blue_bg}" stroke="{blue_border}"/>
          <text x="212" y="17" fill="{blue_text}" font-size="9" font-weight="800" text-anchor="middle">F1 = 70.65%</text>
        </g>
        <!-- Row 3: FPR -->
        <g transform="translate(0, 64)">
          <rect width="280" height="22" rx="4" fill="#f8fafc" stroke="#cbd5e1"/>
          <text x="140" y="15" fill="{text_dark}" font-size="8.5" font-weight="700" text-anchor="middle">FPR = 0.3041% (Low False Alarm Rate)</text>
        </g>
      </g>
      
      <text x="152" y="132" fill="{text_muted}" font-size="7.5" text-anchor="middle">Next-Event Top-1 = 90.05% | Block-Level Top-5 = 98.53%</text>
    </g>

    <!-- Arrow Evaluation -> Error Analysis -->
    <line x1="1704" y1="326" x2="1704" y2="350" stroke="#334155" stroke-width="1.8" marker-end="url(#arr-dark)"/>

    <!-- Process: Error Analysis (Confusion Matrix) -->
    <g transform="translate(1552, 352)">
      <rect width="304" height="114" rx="8" fill="#ffffff" stroke="{border_subtle}" stroke-width="1.2" filter="url(#shadow)"/>
      
      <rect x="1" y="1" width="302" height="22" rx="7" fill="#f1f5f9"/>
      <text x="152" y="16" fill="{text_dark}" font-size="9" font-weight="800" text-anchor="middle" letter-spacing="0.6">ERROR ANALYSIS (CONFUSION MATRIX)</text>

      <!-- 2x2 Matrix -->
      <g transform="translate(12, 32)">
        <!-- Row 1 -->
        <g transform="translate(0, 0)">
          <rect width="136" height="28" rx="4" fill="{green_bg}" stroke="{green_border}"/>
          <text x="68" y="18" fill="{green_text}" font-size="9.5" font-weight="800" text-anchor="middle">TP = 10,115</text>

          <rect x="144" y="0" width="136" height="28" rx="4" fill="{red_bg}" stroke="{red_border}"/>
          <text x="212" y="18" fill="{red_text}" font-size="9.5" font-weight="800" text-anchor="middle">FP = 1,683</text>
        </g>
        <!-- Row 2 -->
        <g transform="translate(0, 34)">
          <rect width="136" height="28" rx="4" fill="{green_bg}" stroke="{green_border}"/>
          <text x="68" y="18" fill="{green_text}" font-size="9.5" font-weight="800" text-anchor="middle">TN = 551,683</text>

          <rect x="144" y="0" width="136" height="28" rx="4" fill="{amber_bg}" stroke="{amber_border}"/>
          <text x="212" y="18" fill="{amber_text}" font-size="9.5" font-weight="800" text-anchor="middle">FN = 6,723</text>
        </g>
      </g>
      <text x="152" y="106" fill="{text_muted}" font-size="8" text-anchor="middle">Target diagnostic focus on False Negatives &amp; False Positives</text>
    </g>

    <!-- Branch Arrows from Error Analysis down to FN & FP -->
    <path d="M 1624 466 L 1624 496" fill="none" stroke="#d97706" stroke-width="1.5" marker-end="url(#arr-amber)"/>
    <path d="M 1784 466 L 1784 496" fill="none" stroke="#dc2626" stroke-width="1.5" marker-end="url(#arr-red)"/>

    <!-- Process: False Negative Analysis (Amber) -->
    <g transform="translate(1552, 498)">
      <rect width="304" height="66" rx="8" fill="{amber_bg}" stroke="{amber_border}" stroke-width="1.2" filter="url(#shadow)"/>
      <text x="14" y="20" fill="{amber_text}" font-size="10.5" font-weight="800">False Negative Analysis</text>
      <text x="14" y="36" fill="{text_dark}" font-size="9" font-weight="700">➔ Identify why anomalies are missed</text>
      <text x="14" y="52" fill="{text_muted}" font-size="8">Investigate subtle anomalous traces accepted in Top-K (FN=6,723)</text>
    </g>

    <!-- Process: False Positive Analysis (Red) -->
    <g transform="translate(1552, 574)">
      <rect width="304" height="66" rx="8" fill="{red_bg}" stroke="{red_border}" stroke-width="1.2" filter="url(#shadow)"/>
      <text x="14" y="20" fill="{red_text}" font-size="10.5" font-weight="800">False Positive Analysis</text>
      <text x="14" y="36" fill="{text_dark}" font-size="9" font-weight="700">➔ Identify why normal blocks are flagged</text>
      <text x="14" y="52" fill="{text_muted}" font-size="8">Investigate rare normal transitions misclassified as anomalies (FP=1,683)</text>
    </g>

    <!-- Merge Arrow from FN/FP to Next Step -->
    <path d="M 1704 640 L 1704 666" fill="none" stroke="#2563eb" stroke-width="2" marker-end="url(#arr-blue)"/>

    <!-- Process: Next Step: Guide Model Improvement -->
    <g transform="translate(1552, 668)">
      <rect width="304" height="96" rx="8" fill="{blue_bg}" stroke="{blue_border}" stroke-width="1.8" filter="url(#shadow)"/>
      <rect x="1" y="1" width="302" height="22" rx="7" fill="#dbeafe"/>
      <text x="152" y="16" fill="{blue_text}" font-size="9" font-weight="800" text-anchor="middle" letter-spacing="0.6">ACTIONABLE NEXT STEP</text>
      
      <text x="152" y="42" fill="{blue_text}" font-size="11" font-weight="800" text-anchor="middle">“Next Step:</text>
      <text x="152" y="58" fill="{blue_text}" font-size="11.5" font-weight="800" text-anchor="middle">Use Error Analysis to Guide</text>
      <text x="152" y="74" fill="{blue_text}" font-size="11.5" font-weight="800" text-anchor="middle">Model Improvement”</text>
      <text x="152" y="90" fill="{text_muted}" font-size="8" text-anchor="middle">Threshold calibration, Top-K tuning &amp; re-weighting</text>
    </g>

    <!-- Arrow Next Step -> END Node -->
    <line x1="1704" y1="764" x2="1704" y2="796" stroke="#334155" stroke-width="2" marker-end="url(#arr-dark)"/>

    <!-- Node: END (Dark rounded box / pill) -->
    <g transform="translate(1552, 798)">
      <rect width="304" height="74" rx="12" fill="#0f172a" stroke="#38bdf8" stroke-width="1.5" filter="url(#shadow)"/>
      
      <rect x="117" y="10" width="70" height="18" rx="9" fill="#38bdf8"/>
      <text x="152" y="23" fill="#0f172a" font-size="10" font-weight="800" text-anchor="middle">END</text>
      
      <text x="152" y="46" fill="#ffffff" font-size="11" font-weight="800" text-anchor="middle">“Deeper Understanding of</text>
      <text x="152" y="62" fill="#38bdf8" font-size="11" font-weight="800" text-anchor="middle">HDFS Log Behavior”</text>
    </g>

    <!-- Lane 5 Bottom Note -->
    <g transform="translate(1552, 980)">
      <rect width="304" height="48" rx="6" fill="#f1f5f9" stroke="#cbd5e1"/>
      <text x="152" y="20" fill="#be185d" font-size="8.5" font-weight="700" text-anchor="middle">CONTINUOUS ITERATION LOOP</text>
      <text x="152" y="36" fill="{text_dark}" font-size="9" font-weight="600" text-anchor="middle">Findings actively feedback to refine model</text>
    </g>
  </g>

  <!-- ================= FOOTER / METADATA ================= -->
  <g id="footer" transform="translate(40, 1060)">
    <text x="0" y="8" fill="#64748b" font-size="8.5" font-weight="500">Project: Post-Review Progress — HDFS Log Analysis &amp; DeepLog • Technical Swimlane Flowchart • 5 Vertical Lanes</text>
    <text x="1840" y="8" fill="#64748b" font-size="8.5" font-weight="500" text-anchor="end">Ground Truth: 575,061 Blocks Validated • Next-Event Top-1: 90.05% • Block-Level Top-5: 98.53% (Preserved Strict Distinction)</text>
  </g>
</svg>"""
    return svg

def main():
    svg_content = generate_svg()
    
    # Validate XML
    try:
        ET.fromstring(svg_content)
        print("✅ Swimlane Flowchart SVG XML is 100% valid!")
    except ET.ParseError as e:
        print(f"❌ XML Parse Error: {e}")
        raise
    
    with open(SVG_FILE, "w", encoding="utf-8") as f:
        f.write(svg_content)
    print(f"✅ Generated Swimlane Flowchart SVG: {SVG_FILE} ({len(svg_content)} bytes)")

    # HTML Viewer with responsive scaling, export options, and full-screen presentation
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Post-Review Progress — HDFS Log Analysis & DeepLog (Technical Swimlane Flowchart)</title>
  <style>
    * {{
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }}
    body {{
      background: #0f172a;
      color: #f8fafc;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 10px 16px;
      overflow: hidden;
    }}
    header {{
      width: 100%;
      max-width: 1920px;
      height: 48px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 0 16px;
      margin-bottom: 8px;
      background: rgba(30, 41, 59, 0.9);
      border: 1px solid #334155;
      border-radius: 8px;
      backdrop-filter: blur(8px);
      flex-shrink: 0;
    }}
    .title-group h1 {{
      font-size: 15px;
      font-weight: 700;
      color: #ffffff;
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .badge {{
      background: #0284c7;
      color: #ffffff;
      font-size: 9.5px;
      font-weight: 800;
      padding: 3px 8px;
      border-radius: 4px;
      letter-spacing: 0.6px;
    }}
    .controls {{
      display: flex;
      gap: 10px;
    }}
    .btn {{
      background: #0f172a;
      color: #38bdf8;
      border: 1px solid #334155;
      padding: 5px 12px;
      font-size: 11.5px;
      font-weight: 600;
      border-radius: 6px;
      cursor: pointer;
      transition: all 0.2s ease;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .btn:hover {{
      background: #2563eb;
      color: #ffffff;
      border-color: #2563eb;
    }}
    .svg-container {{
      flex: 1;
      width: 100%;
      max-width: 1920px;
      display: flex;
      justify-content: center;
      align-items: center;
      overflow: hidden;
    }}
    .svg-viewport {{
      width: 100%;
      max-height: calc(100vh - 76px);
      max-width: calc((100vh - 76px) * 16 / 9);
      aspect-ratio: 16 / 9;
      background: #ffffff;
      border-radius: 10px;
      box-shadow: 0 16px 36px rgba(0, 0, 0, 0.5);
      display: flex;
      justify-content: center;
      align-items: center;
      overflow: hidden;
    }}
    .svg-viewport svg {{
      width: 100%;
      height: 100%;
      display: block;
    }}
  </style>
</head>
<body>
  <header>
    <div class="title-group">
      <h1>
        <span class="badge">5-LANE SWIMLANE</span>
        Post-Review Progress — HDFS Log Analysis &amp; DeepLog
      </h1>
    </div>
    <div class="controls">
      <button class="btn" onclick="toggleFullScreen()">⛶ Fullscreen</button>
      <a href="post_review_swimlane_flowchart.svg" download="post_review_swimlane_flowchart.svg" class="btn">⬇ Download SVG</a>
      <button class="btn" onclick="window.print()">🖨 Print / PDF</button>
    </div>
  </header>

  <div class="svg-container">
    <main class="svg-viewport" id="viewport">
      {svg_content}
    </main>
  </div>

  <script>
    function toggleFullScreen() {{
      const elem = document.getElementById("viewport");
      if (!document.fullscreenElement) {{
        elem.requestFullscreen().catch(err => {{
          alert(`Error attempting to enable fullscreen: ${{err.message}}`);
        }});
      }} else {{
        document.exitFullscreen();
      }}
    }}
  </script>
</body>
</html>
"""
    with open(HTML_FILE, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"✅ Generated Interactive HTML: {HTML_FILE}")

if __name__ == "__main__":
    main()
