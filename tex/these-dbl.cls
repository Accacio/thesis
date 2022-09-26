% Cette classe de document est basee sur [report].
% Assurez vous d'avoir installe tous les paquets utiles
%%%%%% Licence%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% la classe LaTeX these-dbl
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PROPOSITION DE TEMPLATE POUR THESE DE DBL par L.YALA (Doctorante
% IRISA-UR1, Juin 2018)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Revision des en-tetes et prise en compte de [twoside]
% Revision de la gestion de la bibliographie avec BiblaTex
% Ajouts de boites et encarts grises
% revision de l'enchainement des chapitres
% Mise aux normes typographiques suivant les Regles Typographiques
% en usage \`{a} l'Imprimerie Nationale
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{these-dbl}[11/07/2021 v5.1]

\LoadClass[twoside,multicol,12pt,a4paper]{report}	% notez le "twoside", essentiellement pour alterner les en-tete et pieds de page gauche et droit si on veut utliser les fancyhead. 

\RequirePackage[utf8]{inputenc} 
\RequirePackage[english,french]{babel} % Last language is the default, use \selectlanguage{english} to switch
\RequirePackage{ifpdf}
\RequirePackage[usenames,dvipsnames]{color} % pour les couleurs
\RequirePackage{textcomp}
\RequirePackage{titlesec}
\RequirePackage{parcolumns}
\RequirePackage{multicol}
\RequirePackage{booktabs}
\RequirePackage[ruled,noend,algo2e,algochapter]{algorithm2e}
\RequirePackage[T1]{fontenc} % permet de sp\'{e}cifier \`{a} LaTeX l'utilisation du codage de caract\`{e}res T1, nouvelle norme LaTeX non utilis\'{e}e par d\'{e}faut pour des raisons de compatibilit\'{e} avec les anciens documents LaTeX.
\RequirePackage{lmodern} % Assurer une bonne impression!

\RequirePackage{tikz} % tikz est utilise pour tracer des boites, par exemple
\RequirePackage{circuitikz}
\RequirePackage{pgfplots}

\RequirePackage{fix-cm}
\RequirePackage{abstract}
\RequirePackage{graphicx} % Pour ins\'{e}rer des images. Utiliser le format jpg pour plus de simplicit\'{e}.

\graphicspath{{./fig/}{./img}{../img}{../../img}}

\RequirePackage{sectsty} % obligatoire, pour jouer avec l'apparence des titres
\RequirePackage{shorttoc} % pour le sommaire, au debut.
\RequirePackage{fancyhdr} % Afin de r\'{e}aliser soi-même les en-têtes et pieds de page, voir chaque d\'{e}but de chapitre.
\RequirePackage{pifont} % Pour utiliser des symboles divers.
\RequirePackage{color}
\RequirePackage{comment}
\RequirePackage{wallpaper}
\RequirePackage[inner=30mm,outer=20mm,top=30mm,bottom=20mm]{geometry}
\RequirePackage{setspace}
\RequirePackage{xargs}
\RequirePackage[author={Accacio}]{pdfcomment}

\RequirePackage{minitoc}

\RequirePackage{enumitem}

\RequirePackage{mathtools}
\RequirePackage{amsmath}
\RequirePackage{amsthm}
\RequirePackage{amssymb}
\RequirePackage{mathrsfs}
\RequirePackage{mathbbol}

\RequirePackage{eucal}

\RequirePackage{subcaption}
\RequirePackage{float}
\RequirePackage{array}

\RequirePackage{xr}
\RequirePackage{subfiles}
\RequirePackage[math]{blindtext}


%Ou bien : \RequirePackage[top=4 cm, bottom=4 cm, left=3 cm, right=3 cm]{geometry} % On peut modifier ces valeurs pour augmenter ou r\'{e}duire les marges. Attention aux en-têtes de chapitres. 
\RequirePackage{epigraph} % \cleartoevenpage
\setlength{\epigraphwidth}{.5\textwidth}

\RequirePackage{setspace} % permet d'utiliser les commandes \spacing, doublespace (double interligne), singlespace (simple interligne) et onehalfspace (un interligne et demi)

\RequirePackage{hyperref} %Utiliser pour une bibliographie interactive + sommaire et TOC interactifs (pour une sortie PDF, par exemple).

\RequirePackage[acronym,toc,nonumberlist]{glossaries}\makeglossaries

\BeforeBeginEnvironment{document}{
\makeatletter
\hypersetup{
   pdfauthor   = {\@author},
   pdfcreator   = {\@author},
   pdftitle    = {\@title},
   pdfkeywords    = {\@keywordsEN},
   pdfsubject = {\@subject},
   colorlinks=true,
   linkcolor=black,
   citecolor=color-ecole,
   urlcolor=linkcolor,
}
}



\RequirePackage{eso-pic} % For \AddToShipoutPicture used for cover backgrounds
\RequirePackage{framed} % utilise pour coloriser les encarts

\onehalfspacing %Interligne 1,5

\RequirePackage{ifthen} % Entrer valeurs bool\'{e}ennes et autres options
\RequirePackage{csquotes} % Assurer les guillemets français
% \frenchspacing
% \FrenchFootnotes


% Define commands to set fonts throughout the document
\newcommand*{\selectfontfrontcover}{\fontfamily{phv}\selectfont}  % Font style used in front cover 
\newcommand*{\selectfontbackcover}{\fontfamily{phv}\selectfont}   % Font style used in back cover 
\newcommand*{\selectfontchapheads}{\fontfamily{phv}\selectfont} % Font style used chapter headings

% Override default font here if you want
%\renewcommand{\rmdefault}{qpl}
%\renewcommand{\sfdefault}{qpl} 



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%% En-tete chap %%%%%%%%%%%%%%%%

\def\thickhrulefill{\leavevmode \leaders \hrule height 1ex \hfill \kern \z@}
\def\@makechapterhead#1{%
  \cleartooddpage[\thispagestyle{empty}]
  \vspace*{-30\p@}%
  {\parindent \z@ \raggedleft \reset@font
            \scshape \@chapapp{} \thechapter
        \par\nobreak
        \interlinepenalty\@M
    \selectfontchapheads \Huge \bfseries #1\par\nobreak
    %\vspace*{1\p@}%
    \hrulefill
    \par\nobreak
    \vskip 50\p@
  }}
\def\@makeschapterhead#1{%
  \cleartooddpage[\thispagestyle{empty}]
 \vspace*{-50\p@}%
  {\parindent \z@ \raggedleft \reset@font
            \scshape \vphantom{\@chapapp{} \thechapter}
        \par\nobreak
        \interlinepenalty\@M
    \selectfontchapheads \Huge \bfseries #1 \par\nobreak
    %\vspace*{1\p@}%
    \hrulefill
    \par\nobreak
    \vskip 30\p@
  }}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%% Tete de parties %%%%%%%%%%%%%%%%


\usetikzlibrary{svg.path}
\usetikzlibrary{graphs,quotes,graphs.standard}
\tikzset{
  graphs/nodes={draw,circle,inner sep=1pt},
  <->/.style={latex-latex},
  ->/.style={-latex},
  <-/.style={latex-},
}

\usetikzlibrary {chains}
\usetikzlibrary{arrows.meta}
\usetikzlibrary{calc,shapes,positioning,intersections}
\usetikzlibrary{overlay-beamer-styles}
 \usetikzlibrary{plotmarks}
  \usetikzlibrary{arrows.meta}
\usetikzlibrary{babel} % to correct problem with babel

  \newcommand{\triskel}{
  \tikz{
  \begin{scope}[scale=.005] % svg from https://commons.wikimedia.org/wiki/File:Hermine_(champ).svg
  \draw[] svg "M4674.56,2710.82
C4680.63,2723.27,4691.68,2735.86,4703.12,2743.68
C4715.32,2752.01,4732.07,2757.87,4746.75,2759.51
C4763.1,2761.33,4782.52,2758.74,4798.07,2753.34
C4815.72,2747.21,4834.48,2735.55,4848.3,2722.96
C4864.05,2708.62,4878.8,2688.36,4888.27,2669.27
C4899.01,2647.61,4906.64,2620.32,4909.37,2596.29
C4912.44,2569.23,4910.43,2537.46,4904.59,2510.85
C4898.07,2481.14,4884.77,2448.25,4869.34,2422.03
C4852.26,2393,4827.17,2362.83,4802.14,2340.29
C4774.63,2315.52,4738.51,2292.05,4704.93,2276.46
C4668.29,2259.45,4623.16,2246.38,4583.17,2240.66
C4539.79,2234.45,4488.86,2234.84,4445.51,2241.2
C4398.74,2248.05,4346.17,2263.92,4303.2,2283.62
C4257.08,2304.76,4207.64,2336.83,4169.19,2369.94
C4128.1,2405.33,4086.78,2452.84,4057.02,2498.18
C4025.33,2546.45,3996.88,2607.13,3979.59,2662.22
C3961.24,2720.67,3949.73,2790.82,3947.95,2852.05
C3946.07,2916.82,3954.43,2991.53,3970.13,3054.39
C3986.68,3120.68,4016.41,3194.17,4050.19,3253.57
C4085.73,3316.04,4136.6,3382.11,4187.59,3432.77
C4241.11,3485.93,4311.09,3538.45,4376.87,3575.37
C4445.75,3614.04,4531.04,3647.49,4607.74,3666.4
C4687.9,3686.16,4783.12,3696.14,4865.64,3693.87
C4951.74,3691.51,5050.27,3675.17,5132.66,3650.05
C5218.46,3623.89,5312.89,3580.3,5388.7,3532.35
C5467.54,3482.49,5550.25,3412.8,5613.1,3343.87
C5678.36,3272.29,5742.08,3179.82,5786.13,3093.56
C5831.81,3004.11,5870.27,2894.19,5890.75,2795.86
C5911.97,2694.02,5920.41,2573.71,5914.07,2469.88
C5907.5,2362.46,5883.19,2240.1,5848.61,2138.19
C5812.87,2032.89,5755.4,1917.49,5693.26,1825.27
C5629.11,1730.07,5540.59,1630.71,5453.7,1555.69
C5364.09,1478.32,5249.12,1403.38,5142.36,1352.22
C5032.35,1299.51,4897.8,1256.05,4777.83,1234.01
C4654.31,1211.31,4508.92,1204.41,4383.77,1214.86
C4255.04,1225.6,4108.84,1257.9,3987.41,1301.96
C3862.6,1347.24,3726.24,1418.59,3617.62,1494.95
C3506.05,1573.37,3390.03,1680.72,3302.85,1785.58
C3213.36,1893.22,3127.2,2030.68,3068.95,2157.96
C3009.2,2288.51,2960.72,2447.7,2937.13,2589.32
C2915.61,2702.08,2873.81,2828.42,2823.45,2931.58
C2774.66,3031.51,2703.46,3138.97,2630.12,3222.57
C2559.16,3303.48,2465.35,3385.71,2375.61,3445.11
C2288.85,3502.52,2180.45,3555.36,2081.64,3587.91
C1986.24,3619.33,1871.82,3640.99,1771.51,3646.15
C1674.78,3651.13,1562.84,3642.18,1468.22,3621.45
C1377.11,3601.49,1275.4,3564.71,1192.8,3521.36
C1113.37,3479.69,1028.4,3419.7,962.838,3358.47
C899.891,3299.69,836.477,3222.53,791.384,3149.14
C748.161,3078.8,709.189,2991.35,686.225,2912.05
C664.252,2836.18,650.572,2745.59,649.619,2666.6
C648.709,2591.17,659.198,2504.31,678.483,2431.38
C696.859,2361.89,728.637,2284.81,764.975,2222.79
C799.519,2163.83,848.299,2101.34,897.436,2053.86
C944.028,2008.84,1004.55,1964.24,1061.57,1933.49
C1115.49,1904.41,1182,1879.3,1241.8,1865.97
C1298.15,1853.4,1364.93,1847.69,1422.57,1851
C1476.72,1854.11,1538.52,1866.14,1589.72,1884.03
C1637.62,1900.77,1690.11,1927.55,1731.48,1956.93
C1770.02,1984.29,1810.05,2021.88,1839.37,2058.96
C1866.56,2093.32,1892.36,2137.23,1908.67,2177.9
C1923.7,2215.4,1934.96,2261.02,1938.54,2301.25
C1941.83,2338.13,1939.56,2381.15,1931.87,2417.36
C1924.87,2450.3,1911.28,2487.12,1894.68,2516.43
C1879.69,2542.89,1857.85,2570.86,1835.38,2591.36
C1815.26,2609.72,1788.76,2627.35,1763.78,2638.22
C1741.61,2647.87,1714.16,2654.91,1690.02,2656.43
C1668.76,2657.78,1643.84,2655.14,1623.54,2648.67
C1605.74,2642.99,1586.25,2632.58,1572.12,2620.35
C1559.68,2609.59,1547.72,2594.07,1541.12,2579
C1535.2,2565.47,1531.9,2548.03,1533.02,2533.3
C1534.06,2519.48,1539.44,2503.62,1547.19,2492.13
C1554.94,2480.64,1560.32,2464.78,1561.37,2450.96
C1562.49,2436.23,1559.19,2418.79,1553.27,2405.26
C1546.67,2390.19,1534.71,2374.67,1522.27,2363.9
C1508.14,2351.68,1488.65,2341.26,1470.84,2335.59
C1450.55,2329.12,1425.63,2326.48,1404.36,2327.82
C1380.23,2329.35,1352.78,2336.39,1330.61,2346.04
C1305.63,2356.91,1279.13,2374.54,1259.01,2392.9
C1236.54,2413.4,1214.7,2441.36,1199.71,2467.83
C1183.11,2497.14,1169.52,2533.95,1162.52,2566.9
C1154.82,2603.11,1152.56,2646.13,1155.84,2683
C1159.43,2723.24,1170.68,2768.86,1185.72,2806.35
C1202.03,2847.02,1227.83,2890.93,1255.02,2925.3
C1284.34,2962.37,1324.37,2999.97,1362.91,3027.33
C1404.28,3056.71,1456.77,3083.49,1504.67,3100.23
C1555.87,3118.12,1617.67,3130.14,1671.82,3133.25
C1729.46,3136.56,1796.24,3130.86,1852.59,3118.29
C1912.38,3104.95,1978.9,3079.85,2032.81,3050.77
C2089.84,3020.02,2150.36,2975.42,2196.95,2930.4
C2246.09,2882.91,2294.87,2820.42,2329.41,2761.47
C2365.75,2699.45,2397.53,2622.37,2415.91,2552.88
C2435.19,2479.95,2445.68,2393.09,2444.77,2317.66
C2443.82,2238.67,2430.14,2148.08,2408.16,2072.21
C2385.2,1992.9,2346.23,1905.45,2303,1835.11
C2257.91,1761.73,2194.5,1684.57,2131.55,1625.78
C2065.99,1564.56,1981.02,1504.57,1901.59,1462.89
C1818.99,1419.55,1717.28,1382.77,1626.16,1362.8
C1531.55,1342.07,1419.61,1333.12,1322.87,1338.1
C1222.56,1343.27,1108.15,1364.92,1012.75,1396.35
C913.941,1428.9,805.534,1481.74,718.783,1539.15
C629.041,1598.54,535.229,1680.78,464.264,1761.68
C390.933,1845.28,319.733,1952.75,270.941,2052.68
C220.574,2155.84,178.78,2282.17,157.255,2394.93
C135.057,2511.22,127.642,2648.26,136.719,2766.3
C146.072,2887.92,175.706,3026.18,216.605,3141.1
C258.71,3259.41,325.43,3388.78,397.049,3491.94
C470.719,3598.04,571.787,3708.51,670.66,3791.64
C772.286,3877.09,902.253,3959.51,1022.69,4015.4
C1146.39,4072.8,1297.37,4119.61,1431.78,4142.68
C1569.73,4166.36,1731.86,4172.24,1871.21,4159.05
C2014.15,4145.53,2176.25,4107.92,2310.69,4057.53
C2419.11,4019.79,2549.42,3992.81,2663.94,3984.86
C2774.87,3977.15,2903.54,3985.07,3012.61,4006.78
C3118.15,4027.79,3236.28,4067.91,3332.58,4115.93
C3425.68,4162.36,3525.65,4229.82,3603.24,4299.11
C3678.15,4366.02,3754.11,4454.28,3808.74,4538.57
C3861.42,4619.85,3909.64,4721.27,3939,4813.58
C3967.27,4902.47,3986.26,5008.94,3990.03,5102.14
C3993.65,5191.77,3984.19,5295.35,3963.94,5382.74
C3944.5,5466.65,3909.39,5560.15,3868.39,5635.89
C3829.08,5708.49,3772.83,5785.97,3715.64,5845.51
C3660.91,5902.48,3589.3,5959.62,3521.37,5999.94
C3456.5,6038.44,3376.03,6072.79,3303.24,6092.55
C3233.87,6111.38,3151.22,6122.4,3079.34,6121.94
C3011.01,6121.5,2932.51,6110.5,2866.82,6091.69
C2804.53,6073.85,2735.65,6043.74,2680.5,6009.73
C2628.36,5977.57,2573.36,5932.52,2531.92,5887.41
C2492.85,5844.89,2454.52,5789.91,2428.57,5738.34
C2404.19,5689.89,2383.7,5630.35,2373.6,5577.07
C2364.14,5527.22,2361.09,5468.37,2365.85,5417.85
C2370.27,5370.79,2382.82,5317.33,2400.26,5273.4
C2416.43,5232.67,2441.56,5188.37,2468.62,5153.91
C2493.58,5122.14,2527.46,5089.59,2560.51,5066.37
C2590.8,5045.08,2629.19,5025.54,2664.39,5014.1
C2696.43,5003.69,2735.11,4997.05,2768.79,4996.77
C2799.2,4996.51,2834.34,5001.45,2863.33,5010.66
C2889.29,5018.9,2917.81,5033.04,2939.71,5049.23
C2959.16,5063.61,2978.98,5083.86,2992.37,5104
C3004.16,5121.74,3014.33,5144.64,3018.88,5165.46
C3022.87,5183.71,3023.59,5205.8,3020.07,5224.15
C3016.97,5240.31,3009.51,5258.43,2999.76,5271.67
C2991,5283.57,2977.55,5295.14,2964.23,5301.54
C2951.74,5307.54,2935.31,5310.81,2921.49,5309.85
C2907.66,5308.88,2891.24,5312.15,2878.75,5318.16
C2865.43,5324.56,2851.98,5336.13,2843.22,5348.02
C2833.46,5361.27,2826,5379.39,2822.9,5395.55
C2819.39,5413.9,2820.11,5435.98,2824.1,5454.24
C2828.64,5475.05,2838.81,5497.95,2850.61,5515.7
C2864,5535.83,2883.82,5556.08,2903.26,5570.46
C2925.16,5586.66,2953.68,5600.79,2979.64,5609.04
C3008.63,5618.25,3043.77,5623.18,3074.19,5622.93
C3107.87,5622.65,3146.55,5616.01,3178.58,5605.6
C3213.79,5594.16,3252.18,5574.61,3282.46,5553.33
C3315.52,5530.1,3349.4,5497.55,3374.35,5465.78
C3401.42,5431.32,3426.54,5387.02,3442.71,5346.3
C3460.16,5302.36,3472.7,5248.9,3477.13,5201.84
C3481.88,5151.32,3478.83,5092.48,3469.38,5042.62
C3459.27,4989.34,3438.79,4929.8,3414.41,4881.36
C3388.45,4829.78,3350.12,4774.81,3311.06,4732.29
C3269.61,4687.17,3214.62,4642.12,3162.48,4609.97
C3107.33,4575.96,3038.45,4545.84,2976.16,4528.01
C2910.47,4509.19,2831.96,4498.19,2763.63,4497.76
C2691.76,4497.3,2609.11,4508.32,2539.74,4527.15
C2466.94,4546.91,2386.47,4581.26,2321.6,4619.76
C2253.67,4660.08,2182.06,4717.22,2127.34,4774.19
C2070.14,4833.73,2013.89,4911.2,1974.59,4983.81
C1933.59,5059.55,1898.47,5153.05,1879.03,5236.95
C1858.79,5324.34,1849.33,5427.92,1852.95,5517.55
C1856.71,5610.76,1875.71,5717.23,1903.98,5806.12
C1933.33,5898.42,1981.55,5999.85,2034.23,6081.13
C2088.86,6165.42,2164.82,6253.68,2239.74,6320.58
C2317.33,6389.88,2417.3,6457.34,2510.39,6503.76
C2606.7,6551.79,2724.82,6591.91,2830.37,6612.92
C2939.43,6634.62,3068.1,6642.55,3179.04,6634.84
C3293.56,6626.88,3423.87,6599.91,3532.28,6562.17
C3644.09,6523.25,3766.48,6461.15,3864.16,6394.27
C3964.82,6325.36,4069.73,6230.57,4148.81,6137.69
C4230.22,6042.07,4308.89,5919.6,4362.42,5806
C4417.48,5689.15,4462.61,5546.38,4485.17,5419.19
C4508.35,5288.46,4514.75,5134.69,4502.93,5002.45
C4490.79,4866.61,4455.84,4712.46,4408.62,4584.53
C4360.15,4453.21,4284.18,4309.87,4203.08,4195.78
C4119.9,4078.75,4006.27,3957.18,3895.42,3865.94
C3808.52,3790.92,3720.01,3691.56,3655.86,3596.36
C3593.71,3504.14,3536.24,3388.74,3500.51,3283.44
C3465.93,3181.53,3441.61,3059.17,3435.05,2951.75
C3428.7,2847.92,3437.14,2727.61,3458.36,2625.77
C3478.85,2527.44,3517.3,2417.52,3562.98,2328.07
C3607.03,2241.81,3670.76,2149.34,3736.02,2077.76
C3798.87,2008.83,3881.58,1939.14,3960.41,1889.28
C4036.22,1841.33,4130.66,1797.74,4216.46,1771.58
C4298.84,1746.46,4397.37,1730.12,4483.47,1727.76
C4566,1725.49,4661.22,1735.47,4741.38,1755.23
C4818.08,1774.14,4903.37,1807.59,4972.25,1846.26
C5038.03,1883.18,5108.01,1935.7,5161.52,1988.86
C5212.51,2039.52,5263.38,2105.59,5298.92,2168.06
C5332.71,2227.46,5362.43,2300.95,5378.99,2367.24
C5394.68,2430.1,5403.05,2504.81,5401.17,2569.58
C5399.39,2630.81,5387.87,2700.96,5369.53,2759.41
C5352.23,2814.5,5323.79,2875.19,5292.1,2923.45
C5262.33,2968.79,5221.02,3016.3,5179.92,3051.69
C5141.48,3084.8,5092.04,3116.87,5045.91,3138.01
C5002.95,3157.71,4950.37,3173.58,4903.61,3180.43
C4860.25,3186.79,4809.32,3187.18,4765.95,3180.97
C4725.96,3175.25,4680.83,3162.18,4644.18,3145.17
C4610.61,3129.58,4574.49,3106.11,4546.98,3081.34
C4521.94,3058.8,4496.86,3028.63,4479.77,2999.6
C4464.34,2973.38,4451.05,2940.49,4444.53,2910.78
C4438.69,2884.17,4436.67,2852.4,4439.75,2825.34
C4442.48,2801.31,4450.1,2774.02,4460.85,2752.36
C4470.31,2733.27,4485.06,2713.01,4500.82,2698.67
C4514.63,2686.08,4533.4,2674.42,4551.05,2668.29
C4566.59,2662.89,4586.01,2660.3,4602.36,2662.12
C4617.05,2663.76,4633.79,2669.62,4645.99,2677.95
C4657.44,2685.77,4668.48,2698.36,4674.56,2710.82Z";
    \end{scope}
}
}

  \newcommand{\hermine}{
  \tikz{
    \begin{scope}[shift={(2cm,0)},rotate=180,scale=.5] % svg from https://commons.wikimedia.org/wiki/File:Hermine_(champ).svg
    \draw[fill] svg "M -677.03085,9.87064 C -672.85574,15.04988 -684.53088,17.11095 -678.23623,22.01331 C -675.55166,23.67789 -669.28063,20.0745 -674.78974,15.64588 C -675.50156,15.26672 -672.31188,12.64853 -677.03085,9.87064 z";
    \draw[fill] svg "M -670.0662,22.14958 C -671.11067,22.24789 -672.11728,22.89249 -673.06767,24.5372 C -674.25962,27.29458 -669.01807,32.07701 -665.49063,26.19762 C -665.24103,25.48008 -661.73978,27.77808 -659.80279,22.96348 C -663.12343,26.45183 -666.5197,22.28773 -669.60305,22.14963 C -669.75387,22.14287 -669.917,22.13554 -670.0662,22.14958 z";
    \draw[fill] svg "M -682.97894,22.19914 C -686.06235,22.33667 -689.45943,26.51702 -692.77936,23.02803 C -690.84333,27.843 -687.30454,25.54569 -687.05509,26.26327 C -683.52879,32.14335 -678.32337,27.36193 -679.51478,24.60432 C -680.60058,22.72445 -681.77241,22.14533 -682.97894,22.19914 z";
    \draw[fill] svg "M -676.03196,25.55305 C -678.7484,37.88993 -683.80019,46.62352 -689.29934,52.90266 C -687.01361,52.27607 -684.42856,50.49932 -681.94407,48.94045 C -683.4637,52.17543 -685.00131,54.97666 -686.52089,57.83117 C -685.61372,57.52375 -682.72332,54.6411 -678.81329,50.41415 C -678.20092,56.22577 -677.36589,58.97212 -676.12821,62.91283 C -675.14218,58.97863 -674.42731,56.29194 -673.64448,50.31303 C -671.28774,52.98695 -669.01605,55.67485 -665.82718,57.96865 L -670.29109,48.77223 C -667.97937,50.2502 -665.40668,52.18202 -662.95514,53.00682 C -668.76732,46.41323 -673.68621,37.89174 -676.03196,25.55305 z";
    \end{scope}
}}
\newcommand{\chapterEndOrnament}{\vfill \noindent \hrulefill \vfill \vfill}


\let\originalpart=\part%
\def\part{\cleartooddpage[\thispagestyle{empty}]\originalpart}
\newcommand*{\jolipart}[1]{%
	\begin{center}
		\begin{Huge}\color{black}
			#1
		\end{Huge}
		\end{center}
		\vspace{1cm}
	\begin{center}
  \noindent \hrulefill
	\end{center}
	}

\parttitlefont{\jolipart}

\partnumberfont{\sc\normalsize}



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%% REGLAGE DE LA BIBLIOGRAPHIE %%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%% ATTENTION, on utilise Biblatex %%%%%%%%%%%%%

% Les dernieres versions de biblatex pr\'{e}conisent l'option backend=biber
% pour profiter des derni\`{e}res nouveaut\'{e}s. Si vous choisissez cette
% option, il vous faut simplement compiler avec $biber nomdufichier (sans le
% .tex). Pour utiliser bibtex, remplacer par backend=bibtex.
\RequirePackage[hyperref,backend=biber,
  % Exemples de styles: alphabetic, ieee, nature, numeric, verbose-trad1 (en utilisant \footcite{}).
  % https://www.overleaf.com/learn/latex/Biblatex_bibliography_styles
style=alphabetic,
backref=true,
backrefstyle=three,
  ]{biblatex}
% Style biblatex avec gestion des mots cl\'{e}s des entr\'{e}es bibliographiques
% + les abbr\'{e}viations idem, ibidem, op cit. Pr\'{e}voir un fichier bib
% contenant ces mots-cl\'{e}s, ici primary et secondary.
\defbibheading{primary}{\section*{Sources primaires}}
\defbibheading{secondary}{\section*{Sources secondaires}}

\RequirePackage{microtype}

%%%% Ajustements pour la bibliographie

\renewcommand*{\newunitpunct}{\addcomma\space} % Virgule entre les parties d'une reference (merci a Josquin Debaz)

%\DeclareFieldFormat[article]{volume}{\textbf{#1}}  %Le numero de volume en gras
\DeclareFieldFormat[article]{number}{\textit{#1}} %Le numero dans le volume en italique
%\DeclareFieldFormat{pages}{page(s): #1} % page(s) en toutes lettres, si on veut...

% Locutions latines en italique (comme ibid, loc.cit. , etc.) Merci \`{a} Josquin Debaz
\providecommand{\mkibid}{}
\renewcommand{\mkibid}[1]{\emph{#1}}

% Et pour mettre le in en italique dans la ref\'{e}rence biblio (merci encore \`{a} Josquin Debaz)
\DefineBibliographyStrings{english}{%
        in = {\emph{in}}%
}



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%% Interligne simple pour citations et vers %%%%%%%%%%%%%%%%%%
%% Environnements: QUOTE, QUOTATION, VERSE
% Copyright (C) 2002 Mike Nolta <mrnolta@princeton.edu>, GPL V. 2.0 and
% later version. http://www.physics.princeton.edu/cosmology/computing/PrincetonThesis.cls
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\let\orig@quote\quote
\let\endorig@quote\endquote
\renewenvironment*{quote}
	{\begin{orig@quote}\begin{singlespace}}
	{\end{singlespace}\end{orig@quote}}

\let\old@quotation\quotation
\let\endold@quotation\endquotation
\renewenvironment*{quotation}
	{\begin{old@quotation}\begin{singlespace}}
        {\end{singlespace}\end{old@quotation}}

\let\old@verse\verse
\let\endold@verse\endverse
\renewenvironment*{verse}
	{\begin{old@verse}\begin{singlespace}}
        {\end{singlespace}\end{old@verse}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%% VARIABLES PAGE DE GARDE %%%%%%%%%%%%%%%%

%%%%% Dossier contenant les info de l'ecole doctorale
\newcommand*{\direcole}[1]{\gdef\vdirecole{./covers/ecoles-doctorales/#1}}
\direcole{}

%%%%% Nom ecole, une variable par ligne
\newcommand{\nomecoleA}[1]{\gdef\@nomecoleA{#1}}
\nomecoleA{}
\newcommand{\nomecoleB}[1]{\gdef\@nomecoleB{#1}}
\nomecoleB{}

%%%%% Numero ecole doctorale
\newcommand{\numeroecole}[1]{\gdef\@numeroecole{#1}}
\numeroecole{}

%%%% Etablissement delivrant le diplome, une variable par ligne
\newcommand{\nometablissementA}[1]{\gdef\vnometablissementA{#1}}
\nometablissementA{}
\newcommand{\nometablissementB}[1]{\gdef\vnometablissementB{#1}}
\nometablissementB{}
\newcommand{\nometablissementC}[1]{\gdef\vnometablissementC{#1}}
\nometablissementC{}
\newcommand{\nometablissementD}[1]{\gdef\vnometablissementD{#1}}
\nometablissementD{}
\newcommand{\nometablissementE}[1]{\gdef\vnometablissementE{#1}}
\nometablissementE{}

%%%% Logos etablissement delivrant le diplome, supporte deuble affiliation
\newcommand*{\logoetablissementA}[1]{\gdef\vlogoetablissementA{#1}}
\logoetablissementA{}
\newcommand*{\logoetablissementB}[1]{\gdef\vlogoetablissementB{#1}}
\logoetablissementB{}

%%%% Hauteur des logos, variable selon les (double) affiliations
\newlength{\vhauteurlogoecole}
\newcommand*{\hauteurlogoecole}[1]{\setlength{\vhauteurlogoecole}{#1}}
\hauteurlogoecole{2.3cm}
\newcommand*{\hauteurlogoetablissementA}[1]{\gdef\vhauteurlogoetablissementA{#1}}
\hauteurlogoetablissementA{}
\newcommand*{\hauteurlogoetablissementB}[1]{\gdef\vhauteurlogoetablissementB{#1}}
\hauteurlogoetablissementB{3.0cm}

%%%% Eventuel sous-titre
\newcommand{\lesoustitre}[1]{\gdef\@lesoustitre{#1}}
\lesoustitre{}

%%%% Discipline
\newcommand{\discipline}[1]{\gdef\@discipline{#1}}
\discipline{}

\newcommand{\subject}[1]{\gdef\@subject{#1}}
\subject{}

%%%%% Sp\'{e}cialit\'{e}
\newcommand{\spec}[1]{\gdef\@spec{#1}}
\spec{}

%%% Ville de soutenance
\newcommand{\lieu}[1]{\gdef\@lieu{#1}}
\lieu{}

%%% Unite de recherche: laboratoire / department / unit\'{e}
\newcommand{\uniterecherche}[1]{\gdef\@uniterecherche{#1}}
\uniterecherche{}

%%% Num\'{e}ro de la th\`{e}se
\newcommand{\numthese}[1]{\gdef\@numthese{#1}}
\numthese{}



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%% PAGE DE GARDE %%%%%%%%%%%%%%%%
\def\tand{&}
\newcount\@examiner\@examiner0
\newcommand\examiner[3]{%
  \global\@namedef{Examiner:\expandafter\the\@examiner}{#1 & #2, #3}
  \global\advance\@examiner by 1
}
\newcount\@invited\@invited0
\newcommand\invited[3]{%
  \global\@namedef{Invited:\expandafter\the\@invited}{#1 & #2, #3}
  \global\advance\@invited by 1
}
\newcount\@director\@director0
\newcommand\director[3]{%
  \global\@namedef{Director:\expandafter\the\@director}{#1 & #2, #3}
  \global\advance\@director by 1
}

\newcount\@reporter\@reporter0
\newcommand\reporter[3]{%
  \global\@namedef{Reporter:\expandafter\the\@reporter}{#1 & #2, #3}
  \global\advance\@reporter by 1
}

\newcommand\president[3]{\gdef\@president{#1 & #2, #3}}

\newcolumntype{y}[1]{>{\raggedright\let\newline\\\arraybackslash\hspace{0pt}}p{#1}}

%%%% Jury
\newcommand{\jury}[1]{\gdef\@jury{#1}}
\jury{

  \newcounter{it}
\ifnum0=\the\@reporter
\else
{\normalTwelve \textbf{Rapporteurs avant soutenance :}}\\[.2cm]
\footnotesizeTwelve
\quad
\begin{tabular}{@{}y{75pt}y{90pt}l}
  \setcounter{it}{0}
  \@whilenum\theit<\@reporter\do{%
  \tand \protect\csname Reporter:\theit \endcsname
  \stepcounter{it}%
  \ifnum\theit<\@reporter
    \\
  \fi
  }%
\end{tabular}\\[.2cm]
\fi

{\normalTwelve \textbf{Composition du Jury :}}\\[.2cm]
\footnotesizeTwelve
\quad

\begin{tabular}{@{}y{75pt}y{90pt}l}
  Pr\'{e}sident : & \@president\\
  \ifnum0=\the\@examiner
  \errmessage{NO EXAMINERS, at least one examiner is needed}
  \fi
  \setcounter{it}{0}%
  Examinateurs : &
  \protect\csname Examiner:\theit \endcsname  \\
  \stepcounter{it}%
  \@whilenum\theit<\@examiner\do{%
  \tand \protect\csname Examiner:\theit \endcsname  \\
  \stepcounter{it}%
  }%
                   % \end{tabular}\\
  Dir. de thèse : &
                   \setcounter{it}{0}%
                    \protect\csname Director:\theit \endcsname  \\

  \ifnum1<\the\@director
  \stepcounter{it}%
  Co-dir. de thèse : &
  \protect\csname Director:\theit \endcsname  \\
  \stepcounter{it}%
  \@whilenum\theit<\@director\do{%
  \tand \protect\csname Director:\theit \endcsname
  \stepcounter{it}%
  \ifnum\theit<\@director
  \\
  \fi
  }%
  \fi
\end{tabular}\\[.2cm]

\ifnum0=\the\@invited
\else
{\normalTwelve \textbf{Invit\'{e}(s) :}}\\[.2cm]
\footnotesizeTwelve
\quad
\begin{tabular}{@{}y{75pt}y{90pt}l}
  \setcounter{it}{0}%
  \setcounter{it}{0}%
  \@whilenum\theit<\@invited\do{%
  \tand \protect\csname Invited:\theit \endcsname  \\
  \stepcounter{it}%
  }%
\end{tabular}
  \fi
}



% Define some font sizes specific to the covers, supposed to be in 12pt
\newcommand{\HugeTwelve}{\fontsize{26}{31}\selectfont} % 12pt \Huge
\newcommand{\LARGETwelve}{\fontsize{20.74}{25}\selectfont} % 12pt \LARGE
\newcommand{\LargeTwelve}{\fontsize{16}{19}\selectfont} % 12pt \Large
\newcommand{\largeTwelve}{\fontsize{14.4}{17}\selectfont} % 12pt \large
\newcommand{\normalTwelve}{\fontsize{12}{13.2}\selectfont} % 12pt \normalsize
\newcommand{\smallTwelve}{\fontsize{11}{13.5}\selectfont} % 12pt \small
\newcommand{\footnotesizeTwelve}{\fontsize{9.5}{11}\selectfont} % 12pt \footnotesize

% Affiche les logos sur les pages de couverture
% \newcommand{\displayLogos}{%
%   \begin{tikzpicture}[remember picture,overlay,line width=0mm]
%     \node[anchor=north] at (current page.north) {%
%     \parbox{\textwidth}{%
%       % Quand UR1 est l'unique etablissement, il ne faut afficher que son logo
%       {\ifthenelse{\equal{\vlogoetablissementA}{}\and\equal{\vlogoetablissementB}{UR1-noir}}{
%         $\vcenter{\hbox{%
%           \includegraphics[keepaspectratio,height=\vhauteurlogoetablissementB,width=7cm
%           ]{./covers/logos-etablissements/\vlogoetablissementB}%
%         }}$
%       }{%
%         $\vcenter{\hbox{%
%           \includegraphics[keepaspectratio,height=\vhauteurlogoecole,%width=7cm
%           ]{\vdirecole/logo}%
%         }}$
%         \hfill
%         {\if\vlogoetablissementA\empty \else
%           $\vcenter{\hbox{%
%             \includegraphics[keepaspectratio,height=\vhauteurlogoetablissementA,width=7cm
%             ]{./covers/logos-etablissements/\vlogoetablissementA}%
%           }}$
%         \fi
%       }%
%         \hspace{3mm}
%         $\vcenter{\hbox{%
%           \includegraphics[keepaspectratio,height=\vhauteurlogoetablissementB,width=7cm
%           ]{./covers/logos-etablissements/\vlogoetablissementB}%
%         }}$
%       }}%
%     }%
%   };
%   \end{tikzpicture}
%   \par\nobreak
% }
\newcommand{\displayLogos}[1]{%
  \ifthenelse{\equal{\vlogoetablissementA}{}\and\equal{\vlogoetablissementB}{UR1-noir}}
  {
    \node[anchor=center] at (#1) {
      \includegraphics[keepaspectratio,height=\vhauteurlogoetablissementB,width=7cm
      ]{./covers/logos-etablissements/\vlogoetablissementB}%
    };
  }
  {
        \node[align=center,anchor=center] at ($(#1)+(0,-2.5)$) {
          \parbox[c][\vhauteurlogoetablissementB]{\paperwidth}{
            \hfill
            \includegraphics[keepaspectratio,height=\vhauteurlogoecole,%width=7cm
            ]{\vdirecole/logo}%
            \hfill \hfill \hfill
      \includegraphics[keepaspectratio,height=\vhauteurlogoetablissementB,width=7cm]{./covers/logos-etablissements/\vlogoetablissementB}%
            \hfill\hfill}
          };

    % \node[anchor=center] at ($(#1-| .7,0)+(0,-2.5)$) {
    %   \includegraphics[keepaspectratio,height=\vhauteurlogoecole,%width=7cm
    %   ]{\vdirecole/logo}%
    % };

    % \node[anchor=center] at ($(#1-| -2.5,0)+(\textwidth,-2.5)$) {
    %   \includegraphics[keepaspectratio,height=\vhauteurlogoetablissementB,width=7cm]{./covers/logos-etablissements/\vlogoetablissementB}%
    % };

  }
}

%mise en page de la page de garde
\def\maketitle{%
  \thispagestyle{empty}
  \clearpage
    \begin{tikzpicture}[remember picture,overlay,line width=0mm]
        \setlength{\baselineskip}{0.8\baselineskip}
      \selectfontfrontcover % Set font style for front cover page
      \begin{scope}[blend mode = normal]
      % \begin{scope}[blend mode = color]
        % TODO this image should be created using TiKz comands, extract svg paths and use ecole-color to fill paths
        \node[anchor=center,yshift=.22\paperheight] (fond) at (current page.center) {
          \includegraphics[width=\paperwidth,keepaspectratio,trim={0 5pt 0 0}]{\vdirecole/image-fond-garde}
        };
      \end{scope}
      \fill[color-ecole] (-3.625,-14) rectangle ++(\paperwidth,-1);

      \node (logoposition) at (current page.north){};
      \displayLogos{logoposition}
      \node[align=left,anchor=west] at (-2,-2.5) {\HugeTwelve \textsc{Thèse de doctorat de} \\};

      \node[align=left,anchor=west] at (-2,-5.) {
        \parbox{\paperwidth}{{\normalTwelve \if\@nomecoleB\empty ~\\ \else \fi} % To compensate the 2 lines of MathSTIC
          {\largeTwelve \if\vnometablissementA\empty ~ \else \vnometablissementA \fi} \\
          {\largeTwelve \if\vnometablissementB\empty ~ \else \vnometablissementB \fi} \\
          {\largeTwelve \if\vnometablissementC\empty ~ \else \vnometablissementC \fi} \\
          {\largeTwelve \if\vnometablissementD\empty ~ \else \vnometablissementD \fi} \\
          {\largeTwelve \vnometablissementE} \\
          }

      };
      \node[align=left,anchor=west] at (-2,-7.5) {
        \smallTwelve \textsc{\'{E}cole Doctorale \No \@numeroecole}\\
         \textit{\@nomecoleA} \\
         \if\@nomecoleB\empty \else \textit{\@nomecoleB} \\ \fi
         Sp\'{e}cialit\'{e} : \textit{\@spec}
      };
      \node[align=left,anchor=north west] at (-2,-9.5) {
        \hspace{0.6cm}{\normalTwelve Par}\\[.2cm]
        \hspace{0.6cm}{\LARGETwelve \textbf{\@author}}\\[.5cm]
        {\LargeTwelve \textbf{\@title}} \\[0.25cm]
        {\largeTwelve \@lesoustitre} \\[0.5cm]
        	 \smallTwelve
           \textbf{Th\`{e}se pr\'{e}sent\'{e}e et soutenue \`{a} \@lieu, le \@date} \\
           \textbf{Unit\'{e} de recherche : \@uniterecherche} \\
           \textbf{\if\@numthese\empty \else Th\`{e}se \No : \@numthese \fi} % Hide line if no number provided
      };

      \node[align=left,anchor=north west] at (-2,-16.5) {
          \@jury
      };

    \end{tikzpicture}


}



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%% QUATRIEME DE COUVERTURE %%%%%%%%%%%%%%%%
\newcommand{\titleEN}[1]{\gdef\@titleEN{#1}}
\titleEN{}

\newcommand{\abstractEN}[1]{\gdef\@abstractEN{#1}}
\abstractEN{}

\newcommand{\keywordsEN}[1]{\gdef\@keywordsEN{#1}}
\keywordsEN{}

\newcommand{\abstractFR}[1]{\gdef\@abstractFR{#1}}
\abstractEN{}

\newcommand{\keywordsFR}[1]{\gdef\@keywordsFR{#1}}
\keywordsFR{}

\newcommand{\titleFR}[1]{\gdef\@titleFR{#1}}
\titleFR{}


\newcommand{\makebackcover}{%
  \cleartooddpage[\thispagestyle{empty}]
  % \newgeometry{inner=30mm,outer=20mm,top=40mm,bottom=20mm}
  \selectfontbackcover
  \thispagestyle{empty}
  \clearpage
  \begin{tikzpicture}[remember picture,overlay,line width=0mm]
    \node (logoposition) at (current page.north){};
    \displayLogos{logoposition}
  \end{tikzpicture}

  \AddToShipoutPicture*{%
    \put(0,0){%
      \parbox[t][\paperheight]{\paperwidth}{%
        \vspace{-29.6cm}
        \includegraphics[width=\paperwidth,height=\paperheight]{\vdirecole/image-fond-dos}%
      }}
  }
  \vspace{1cm}
  {\centering \noindent \textcolor{color-ecole}{\rule{\textwidth}{0.2cm}}}
  \vspace{-1cm}
  \selectlanguage{french}
  \section*{\selectfontbackcover\smallTwelve \textcolor{color-ecole}{Titre : }{\selectfontbackcover\mdseries{\bfseries\@titleFR}}}
  \vspace{-0.2cm}
  \noindent{\smallTwelve \textbf{Mot cl\'{e}s : }\@keywordsFR}

  \vspace{-0.2cm}
  \begin{multicols}{2}
    \begin{spacing}{1}
      \noindent\smallTwelve \textbf{R\'{e}sum\'{e} : } \@abstractFR
    \end{spacing}
  \end{multicols}

  \vspace{1cm}
  {\centering \noindent \textcolor{color-ecole}{\rule{\textwidth}{0.2cm}}}
  \vspace{-1cm}
  \selectlanguage{english}
  \section*{\selectfontbackcover\smallTwelve \textcolor{color-ecole}{Title : }{\selectfontbackcover\mdseries{\bfseries\@titleEN}}}
  \vspace{-0.2cm}
  \noindent{\smallTwelve \textbf{Keywords: }\@keywordsEN}

  \vspace{-0.2cm}
  \begin{multicols}{2}
    \begin{spacing}{1}
      \noindent\smallTwelve \textbf{Abstract: } \@abstractEN
    \end{spacing}
  \end{multicols}
  % \newpage
  % \restoregeometry
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%% EN-TETES PAGES %%%%%%%%%%%%%%%%

%%%%%%%%% Pour suppirmer les entetes et pied de page gênants par exemple juste avant un chapitre sur une page de droite
\newcommand{\clearemptydoublepage}{%
   \newpage{\pagestyle{empty}\cleardoublepage}}
%%%% .... et utiliser la commande \clearemptydoublepage juste avant le \chapter
\let\oldchapter\chapter
\def\chapter{\clearemptydoublepage\oldchapter}

%%%%

\addto\captionsenglish{
  \renewcommand{\contentsname}{Table of Contents}
}

\let\oldglossary\printglossary
\def\printglossary{
  \renewcommand{\glsnamefont}[1]{\makefirstuc{##1}}%
\cleartooddpage[\thispagestyle{empty}]
\oldglossary[style=alttree]\adjustmtc
}

\providecommand{\listacronymsname}{}

\addto\captionsenglish{
  \renewcommand{\listacronymsname}{List of Acronyms}
}
\addto\captionsfrench{
  \renewcommand{\listacronymsname}{Liste des abréviations}
}

\captionsetup{justification=centering}
\renewcommand{\glossarypreamble}{\glsfindwidesttoplevelname[\currentglossary]}
\renewcommand*{\glstreenamebox}[2]{%
\makebox[1.1#1][l]{#2}%
}
\def\listofacronyms{
\cleartooddpage[\thispagestyle{empty}]
% \addcontentsline{toc}{chapter}{\listacronymsname}\adjustmtc
\oldglossary[type=\acronymtype,style=alttree,title=\listacronymsname]\adjustmtc
}

\let\oldlistoftables\listoftables
\def\listoftables{
\cleartooddpage[\thispagestyle{empty}]
\phantomsection\addcontentsline{toc}{chapter}{\listtablename}
\oldlistoftables\adjustmtc
}

\let\oldlistoffigures\listoffigures
\def\listoffigures{
\cleartooddpage[\thispagestyle{empty}]
\phantomsection\addcontentsline{toc}{chapter}{\listfigurename}
\oldlistoffigures\adjustmtc
}



% \cleartooddpage[\thispagestyle{empty}]
% \addcontentsline{toc}{chapter}{\listfigurename}\adjustmtc
% \listoffigures

\fancyhf{}                       % on annule le fancy automatique
   

%%%%%%%%%% Gerer les en tetes dans les frontmatter mainmatter et backmatter
\RequirePackage{geometry}
\RequirePackage{nicefrac,xfrac}
\RequirePackage{etoolbox}
\RequirePackage[framemethod=TikZ]{mdframed}
\RequirePackage{fontawesome}
\RequirePackage{mathabx}
\RequirePackage{accents}
% correct ddots
% shamelessly taken from https://tex.stackexchange.com/a/246695/162943
\makeatletter
\DeclareRobustCommand\vdots{%
  \mathpalette\@vdots{}%
}
\newcommand*{\@vdots}[2]{%
  % #1: math style
  % #2: unused
  \sbox0{$#1\cdotp\cdotp\cdotp\m@th$}%
  \sbox2{$#1.\m@th$}%
  \vbox{%
    \dimen@=\wd0 %
    \advance\dimen@ -3\ht2 %
    \kern.5\dimen@
    % remove side bearings
    \dimen@=\wd2 %
    \advance\dimen@ -\ht2 %
    \dimen2=\wd0 %
    \advance\dimen2 -\dimen@
    \vbox to \dimen2{%
      \offinterlineskip
      \copy2 \vfill\copy2 \vfill\copy2 %
    }%
  }%
}
\DeclareRobustCommand\ddots{%
  \mathinner{%
    \mathpalette\@ddots{}%
    \mkern\thinmuskip
  }%
}
\newcommand*{\@ddots}[2]{%
  % #1: math style
  % #2: unused
  \sbox0{$#1\cdotp\cdotp\cdotp\m@th$}%
  \sbox2{$#1.\m@th$}%
  \vbox{%
    \dimen@=\wd0 %
    \advance\dimen@ -3\ht2 %
    \kern.5\dimen@
    % remove side bearings
    \dimen@=\wd2 %
    \advance\dimen@ -\ht2 %
    \dimen2=\wd0 %
    \advance\dimen2 -\dimen@
    \vbox to \dimen2{%
      \offinterlineskip
      \hbox{$#1\mathpunct{.}\m@th$}%
      \vfill
      \hbox{$#1\mathpunct{\kern\wd2}\mathpunct{.}\m@th$}%
      \vfill
      \hbox{$#1\mathpunct{\kern\wd2}\mathpunct{\kern\wd2}\mathpunct{.}\m@th$}%
    }%
  }%
}
\makeatother



\appto\frontmatter{\pagestyle{fancy}
\renewcommand{\sectionmark}[1]{}
\renewcommand{\chaptermark}[1]{\markboth{\textit{#1}}{}}
\fancyhead[LE,RO]{\small\thepage}
    \fancyhead[RO]{\small\leftmark}   % \rightmark = section courante
    \fancyhead[LE]{\small\leftmark} % \leftmark = chapitre courant
    \fancyfoot[C]{\thepage}               % marque la page au centre
}

\appto\mainmatter{\pagestyle{fancy}
\renewcommand{\sectionmark}[1]{\markright{\textit{\thesection.\ #1}}}
\renewcommand{\chaptermark}[1]{\markboth{\chaptername~\thechapter~--\ \textit{#1}}{}}
\fancyhead[LE,RO]{\small\thepage}
    \fancyhead[RO]{\small\rightmark}   % \rightmark = section courante
    \fancyhead[LE]{\small Partie~\thepart, \leftmark} % \leftmark = chapitre courant
    \fancyfoot[C]{\thepage}               % marque la page au centre
}     

\appto\backmatter{\pagestyle{fancy}
\renewcommand{\sectionmark}[1]{\markright{\thesection.\ #1}}
\renewcommand{\chaptermark}[1]{\markboth{\chaptername~\thechapter~--\ #1}{}}
\fancyhead[LE,RO]{\small\thepage}
    \fancyhead[RO]{}   % \rightmark = section courante
    \fancyhead[LE]{} % \leftmark = chapitre courant
    \fancyfoot[C]{\thepage}               % marque la page au centre

} 



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%% FAIRE DES ENCARTS %%%%%%%%%%%%%%%%
\definecolor{linkcolor}{RGB}{7, 109, 174}
\definecolor{fondtitre}{RGB}{85,85,85}
\definecolor{fonddeboite}{RGB}{232,232,232}
\definecolor{shadecolor}{RGB}{232,232,232}

\DeclareOption{ornaments}{
\renewcommand{\chapterEndOrnament}{\vfill \noindent \hrulefill \raisebox{-15pt}{\hermine} \hrulefill \vfill \vfill}
\renewcommand*{\jolipart}[1]{%
	\begin{center}
		\begin{Huge}\color{black}
			#1
		\end{Huge}
		\end{center}
		\vspace{1cm}
	\begin{center}
  \noindent \hrulefill~\raisebox{-10pt}{\triskel}~\hrulefill
	\end{center}
	}

}

\ProcessOptions

%%%%%%%%%% Boitemagique  %%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand*{\magicbox}[2]{
\begin{center}
\begin{tikzpicture}
% la boite
\node[rectangle,draw=fondtitre!100,fill=fonddeboite!100,inner sep=10pt, inner ysep=20pt,line width=1pt] (mabox)
{
\begin{minipage}{12cm}
#2
\end{minipage}
};
% le titre de la boite
\ifstrempty{#1}{}
{
\node[fill=fondtitre!80, text=white, rectangle] at (mabox.north) {\sffamily\textbf{#1}};
}
\end{tikzpicture}
\end{center}
}

%%%%%% boitesimple %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand*{\simplebox}[1]{%
\magicbox{}
{
  #1
}
}

\newcounter{theorem}[chapter]
\newenvironment{theorem}[1][]{%
\stepcounter{theorem}%
\ifstrempty{#1}%
 {\mdfsetup{%
   frametitle={%
    \tikz[baseline=(current bounding box.east),outer sep=0pt]
    \node[rectangle,fill=fondtitre!80,text=white,anchor=center]
         {\sffamily\textbf{\strut Theorem~\thechapter.\thetheorem}};}}
 }%
{\mdfsetup{%
  frametitle={%
   \tikz[baseline=(current bounding box.east),outer sep=0pt]
   \node[rectangle,fill=fondtitre!80,text=white,anchor=center]
        {\sffamily\textbf{\strut Theorem~\thechapter.\thetheorem:~#1}};}}%
 }%
\mdfsetup{innertopmargin=10pt,linecolor=fondtitre!100,backgroundcolor=fonddeboite!100,%
       linewidth=1pt,topline=true,
       frametitleaboveskip=\dimexpr-\ht\strutbox\relax,}
   \begin{mdframed}[]\relax%
}
{\end{mdframed}}
\renewcommand{\thetheorem}{\arabic{chapter}.\arabic{theorem}}

\newmdenv[innertopmargin=10pt,linecolor=fondtitre!100,backgroundcolor=fonddeboite!100,%
 frametitleaboveskip=\dimexpr-\ht\strutbox\relax,
 linewidth=1pt,topline=true]{infobox}

\definecolor{example_title_background}{HTML}{223322}
\colorlet{example_title_background}{fondtitre!80}
\definecolor{example_background}{HTML}{5E81AC}
% \colorlet{example_background}{example_background!50}

\colorlet{example_background}{fonddeboite!100}
% \colorlet{example_background}{fonddeboite!100}
\newcounter{example}[chapter]
\newenvironment{example}[1][]
{%
\stepcounter{example}%
\ifstrempty{#1}%
 {\begin{infobox}[%
     skipabove={1.2\baselineskip},
     innertopmargin={1.2\baselineskip},
     firstextra={\useasboundingbox (P) rectangle (P); \node[fill=fondtitre!80,text=white,anchor=west,xshift=5mm] at (P-|O)  {\sffamily\textbf{\strut Example~\thechapter.\theexample}};},
     singleextra={\useasboundingbox (P) rectangle (P); \node[fill=fondtitre!80,text=white,anchor=west,xshift=5mm] at (P-|O)  {\sffamily\textbf{\strut Example~\thechapter.\theexample}};},
   ]
 }%
 {\begin{infobox}[%
     skipabove={1.2\baselineskip},
     innertopmargin={1.2\baselineskip},
     firstextra={\useasboundingbox (P) rectangle (P); \node[fill=fondtitre!80,text=white,anchor=west,xshift=5mm] at (P-|O)  {\sffamily\textbf{\strut Example~\thechapter.\theexample:~#1}};},
     singleextra={\useasboundingbox (P) rectangle (P); \node[fill=fondtitre!80,text=white,anchor=west,xshift=5mm] at (P-|O)  {\sffamily\textbf{\strut Example~\thechapter.theexample:~#1}};},
     ]%
   }%
}
{\end{infobox}}
\renewcommand{\theexample}{\arabic{chapter}.\arabic{example}}

\newcounter{remark}[chapter]
\newenvironment{remark}[1][]
{%
\pagebreak[1]
\refstepcounter{remark}%
\begin{infobox}[
  skipabove={1.2\baselineskip},
  innertopmargin={1.2\baselineskip},
  firstextra={\useasboundingbox (P) rectangle (P); \node[fill=fondtitre!80,text=white,anchor=west,xshift=5mm] at (P-|O)  {\sffamily\textbf{\strut Remark~\theremark}};},
  singleextra={\useasboundingbox (P) rectangle (P); \node[fill=fondtitre!80,text=white,anchor=west,xshift=5mm] at (P-|O)  {\sffamily\textbf{\strut Remark~\theremark}};},
 ]
 %https://tex.stackexchange.com/questions/135083/mdframed-does-not-split-correctly-between-pages
}
{
\end{infobox}
}
\renewcommand{\theremark}{\arabic{chapter}.\arabic{remark}}


%%% Switch case in latex
%%% https://tex.stackexchange.com/a/343306
\newcommand\addcase[3]{\expandafter\def\csname\string#1@case@#2\endcsname{#3}}
\newcommand\makeswitch[2][]{%
  \newcommand#2[1]{%
    \ifcsname\string#2@case@##1\endcsname\csname\string#2@case@##1\endcsname\else#1\fi%
  }%
}

%%%% Il faut adapter la taille des logos dans certains cas (e.g., EGAAL, 2 etablissements)
\newcommand\hauteurlogos[3]{
    \hauteurlogoecole{#1}
    \hauteurlogoetablissementA{#2}
    \hauteurlogoetablissementB{#3}
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%% ECOLES DOCTORALES %%%%%%%%%%%%%%%%

%%%% #1: dossier des images, #2: numero ED, #3: couleur ED, #4-#5: nom complet sur plusieurs lignes
\newcommand\addecoledoctorale[5]{\direcole{#1}\numeroecole{#2}\definecolor{color-ecole}{RGB}{#3}\nomecoleA{#4}\nomecoleB{#5}}

\makeswitch[default]\ecoledoctorale{}

\addcase\ecoledoctorale{3M}{\addecoledoctorale
    {3M}
    {596}
    {193,192,183}
    {Mati\`{e}re, Mol\'{e}cules, Mat\'{e}riaux}
    {}
}
\addcase\ecoledoctorale{ALL}{\addecoledoctorale
    {ALL}
    {595}
    {240,209,134}
    {Arts, Lettres, Langues}
    {}
}
\addcase\ecoledoctorale{BS}{\addecoledoctorale
    {BS}
    {605}
    {163,219,208}
    {Biologie, Sant\'{e}}
    {}
}
\addcase\ecoledoctorale{DSP}{\addecoledoctorale
    {DSP}
    {599}
    {188,208,220}
    {Droit et Science politique}
    {}
}
\addcase\ecoledoctorale{EDGE}{\addecoledoctorale
    {EDGE}
    {597}
    {216,178,210}
    {Sciences \'{E}conomiques et sciences De Gestion}
    {}
}
\addcase\ecoledoctorale{EGAAL}{\addecoledoctorale
    {EGAAL}
    {600}
    {146,213,182}
    {\'{E}cologie, G\'{e}osciences, Agronomie et Alimentation}
    {}
    \hauteurlogos{2cm}{2cm}{2cm}
}
\addcase\ecoledoctorale{ELICC}{\addecoledoctorale
    {ELICC}
    {603}
    {249,201,188}
    {\'{E}ducation, Langages, Interactions, Cognition, Clinique}
    {}
    \hauteurlogos{2cm}{2cm}{2cm}
}
\addcase\ecoledoctorale{MathSTIC}{\addecoledoctorale
    {MathSTIC}
    {601}
    {236,115,127}
    {Math\'{e}matiques et Sciences et Technologies}
    {de l'Information et de la Communication}
}
\addcase\ecoledoctorale{SML}{\addecoledoctorale
    {SML}
    {598}
    {162,225,230}
    {Sciences de la Mer et du Littoral}
    {}
}
\addcase\ecoledoctorale{SPI}{\addecoledoctorale
    {SPI}
    {602}
    {159,182,217}
    {Sciences pour l'Ing\'{e}nieur}
    {}
}
\addcase\ecoledoctorale{STT}{\addecoledoctorale
    {STT}
    {604}
    {172,184,192}
    {Soci\'{e}t\'{e}s, temps, territoires}
    {}
}



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%% ETABLISSEMENTS %%%%%%%%%%%%%%%%

%%%% #1 nom du logo, #2-#4: nom complet sur plusieurs lignes
\newcommand\addetablissement[4]{\logoetablissementB{#1}\nometablissementC{#2}\nometablissementD{#3}\nometablissementE{#4}}

\makeswitch[default]\etablissement{}

\addcase\etablissement{CS}{\addetablissement
    {CS}
    {}
    {}
    {CENTRALESUP\'{E}LEC}
}
\addcase\etablissement{ECN}{\addetablissement
    {ECN}
    {}
    {}
    {L'\'{E}COLE CENTRALE DE NANTES}
}
\addcase\etablissement{EHESP}{\addetablissement
    {EHESP}
    {}
    {L'\'{E}COLE DES HAUTES \'{E}TUDES}
    {EN SANT\'{E} PUBLIQUE DE RENNES}
}
\addcase\etablissement{ENIB}{\addetablissement
    {ENIB}
    {}
    {L'\'{E}COLE NATIONALE}
    {D'ING\'{E}NIEURS DE BREST}
}
\addcase\etablissement{ENS}{\addetablissement
    {ENS}
    {}
    {L'\'{E}COLE NORMALE}
    {SUP\'{E}RIEURE RENNES}
}
\addcase\etablissement{ENSA}{\addetablissement
    {ENSA}
    {}
    {L'\'{E}COLE NORMALE SUP\'{E}RIEURE}
    {D'ARCHITECTURE DE NANTES}
}
\addcase\etablissement{ENSAB}{\addetablissement
    {ENSAB}
    {}
    {L'\'{E}COLE NORMALE SUP\'{E}RIEURE}
    {D'ARCHITECTURE DE BRETAGNE}
}
\addcase\etablissement{ENSAI}{\addetablissement
    {ENSAI}
    {}
    {L'\'{E}COLE NATIONALE DE LA STATISTIQUE}
    {ET DE L'ANALYSE DE L'INFORMATION}
}
\addcase\etablissement{ENSCR}{\addetablissement
    {ENSCR}
    {}
    {L'\'{E}COLE NATIONALE SUP\'{E}RIEURE}
    {DE CHIMIE RENNES}
}
\addcase\etablissement{ENSTA}{\addetablissement
    {ENSTA}
    {}
    {L'\'{E}COLE NATIONALE SUP\'{E}RIEURE}
    {DE TECHNIQUES AVANC\'{E}ES BRETAGNE}
}
\addcase\etablissement{IMTA}{\addetablissement
    {IMTA}
    {L'\'{E}COLE NATIONALE SUP\'{E}RIEURE}
    {MINES-T\'{E}L\'{E}COM ATLANTIQUE BRETAGNE}
    {PAYS-DE-LA-LOIRE - IMT ATLANTIQUE}
}
\addcase\etablissement{INSA}{\addetablissement
    {INSA}
    {}
    {L'INSTITUT NATIONAL DES}
    {SCIENCES APPLIQU\'{E}ES RENNES}
}
\addcase\etablissement{InstitutAgro}{\addetablissement
    {InstitutAgro}
    {L'INSTITUT NATIONAL D'ENSEIGNEMENT SUP\'{E}RIEUR}
    {POUR L'AGRICULTURE, L'ALIMENTATION ET}
    {L'ENVIRONNEMENT - ECOLE INTERNE AGROCAMPUS OUEST}
}
\addcase\etablissement{LMU}{\addetablissement
    {LMU}
    {}
    {}
    {LE MANS UNIVERSIT\'{E}}
}
\addcase\etablissement{Oniris}{\addetablissement
    {Oniris}
    {}
    {}
    {ONIRIS}
}
\addcase\etablissement{UA}{\addetablissement
    {UA-couleur}
    {}
    {}
    {L'UNIVERSIT\'{E} D'ANGERS}
}
\addcase\etablissement{UB}{\addetablissement
    {UB}
    {}
    {}
    {L'UNIVERSIT\'{E} DE BREST}
}
\addcase\etablissement{UBO}{\addetablissement
    {UBO}
    {}
    {}
    {L'UNIVERSIT\'{E} DE BRETAGNE OCCIDENTALE}
}
\addcase\etablissement{UBS}{\addetablissement
    {UBS}
    {}
    {}
    {L'UNIVERSIT\'{E} DE BRETAGNE SUD}
}
\addcase\etablissement{UN}{\addetablissement
    {UN-noir}
    {}
    {}
    {L'UNIVERSIT\'{E} DE NANTES}
}
\addcase\etablissement{UR1}{\addetablissement
    {UR1-noir}
    {}
    {}
    {L'UNIVERSIT\'{E} DE RENNES 1}
}
\addcase\etablissement{UR2}{\addetablissement
    {UR2}
    {}
    {}
    {L'UNIVERSIT\'{E} DE RENNES 2}
}


%%%% #1-#2: nom des deux logos, #3-#7: nom complet de la double affiliation sur plusieurs lignes
\newcommand\addpairetablissements[7]{
    \logoetablissementA{#1}
    \logoetablissementB{#2}
    \nometablissementA{#3}
    \nometablissementB{#4}
    \nometablissementC{#5}
    \nometablissementD{#6}
    \nometablissementE{#7}
}

% ALL, STT: UR2-ENSAB
\addcase\etablissement{UR2-ENSAB}{\addpairetablissements
    {ENSAB}
    {UR2}
    {}
    {L'\'{E}COLE NORMALE SUP\'{E}RIEURE}
    {D'ARCHITECTURE DE BRETAGNE}
    {D\'{E}LIVR\'{E}E CONJOINTEMENT AVEC}
    {L'UNIVERSIT\'{E} DE RENNES 2}
    \hauteurlogos{2cm}{1.2cm}{2cm}
}
% BS, DSP, MathSTIC: UR1-UR2
\addcase\etablissement{UR1-UR2}{\addpairetablissements
    {UR2}
    {UR1-noir}
    {}
    {}
    {L'UNIVERSIT\'{E} DE RENNES 2}
    {D\'{E}LIVR\'{E}E CONJOINTEMENT AVEC}
    {L'UNIVERSIT\'{E} DE RENNES 1}
    \hauteurlogos{2cm}{2cm}{2cm}
}
% DSP, EDGE: UR1-EHESP
\addcase\etablissement{UR1-EHESP}{\addpairetablissements
    {EHESP}
    {UR1-noir}
    {}
    {L'UNIVERSIT\'{E} DE RENNES 1}
    {D\'{E}LIVR\'{E}E CONJOINTEMENT AVEC}
    {L'\'{E}COLE DES HAUTES \'{E}TUDES}
    {EN SANT\'{E} PUBLIQUE DE RENNES}
    \hauteurlogos{2cm}{2cm}{2cm}
}
% EGAAL: UA-LMU
\addcase\etablissement{UA-LMU}{\addpairetablissements
    {LMU}
    {UA-couleur}
    {}
    {}
    {LE MANS UNIVERSIT\'{E}}
    {D\'{E}LIVR\'{E}E CONJOINTEMENT AVEC}
    {L'UNIVERSIT\'{E} D'ANGERS}
    \hauteurlogos{2cm}{1cm}{2cm}
}
% MathSTIC: UR1-InstitutAgro
\addcase\etablissement{UR1-InstitutAgro}{\addpairetablissements
    {InstitutAgro}
    {UR1-noir}
    {L'INSTITUT NATIONAL D'ENSEIGNEMENT SUP\'{E}RIEUR}
    {POUR L'AGRICULTURE, L'ALIMENTATION ET}
    {L'ENVIRONNEMENT - ECOLE INTERNE AGROCAMPUS OUEST}
    {D\'{E}LIVR\'{E}E CONJOINTEMENT AVEC}
    {L'UNIVERSIT\'{E} DE RENNES 1}
    \hauteurlogos{1.8cm}{1.3cm}{1.5cm}
}
% SML: UBO-IMTA
\addcase\etablissement{UBO-IMTA}{\addpairetablissements
    {IMTA}
    {UBO}
    {L'\'{E}COLE NATIONALE SUP\'{E}RIEURE}
    {MINES-T\'{E}L\'{E}COM ATLANTIQUE BRETAGNE}
    {PAYS-DE-LA-LOIRE - IMT ATLANTIQUE}
    {D\'{E}LIVR\'{E}E CONJOINTEMENT AVEC}
    {L'UNIVERSIT\'{E} DE BRETAGNE OCCIDENTALE}
    \hauteurlogos{2cm}{1.8cm}{1.8cm}
}
% SPI: ECN-ENSA
\addcase\etablissement{ECN-ENSA}{\addpairetablissements
    {ENSA}
    {ECN}
    {}
    {L'\'{E}COLE NORMALE SUP\'{E}RIEURE}
    {D'ARCHITECTURE DE NANTES}
    {D\'{E}LIVR\'{E}E CONJOINTEMENT AVEC}
    {L'\'{E}COLE CENTRALE DE NANTES}
    \hauteurlogos{2cm}{1.8cm}{1.8cm}
}
% SPI: UBO-ENIB
\addcase\etablissement{UBO-ENIB}{\addpairetablissements
    {ENIB}
    {UBO}
    {}
    {L'\'{E}COLE NATIONALE}
    {D'ING\'{E}NIEURS DE BREST}
    {D\'{E}LIVR\'{E}E CONJOINTEMENT AVEC}
    {L'UNIVERSIT\'{E} DE BRETAGNE OCCIDENTALE}
    \hauteurlogos{2cm}{1.8cm}{1.6cm}
}
% SPI: UN-Oniris
\addcase\etablissement{UN-Oniris}{\addpairetablissements
    {Oniris}
    {UN-noir}
    {}
    {}
    {ONIRIS}
    {D\'{E}LIVR\'{E}E CONJOINTEMENT AVEC}
    {L'UNIVERSIT\'{E} DE NANTES}
    \hauteurlogos{2cm}{2cm}{2cm}
}
% STT: ENSA-UN
\addcase\etablissement{ENSA-UN}{\addpairetablissements
    {ENSA}
    {UN-noir}
    {}
    {L'\'{E}COLE NORMALE SUP\'{E}RIEURE}
    {D'ARCHITECTURE DE NANTES}
    {D\'{E}LIVR\'{E}E CONJOINTEMENT AVEC}
    {L'UNIVERSIT\'{E} DE NANTES}
    \hauteurlogos{2cm}{2cm}{2cm}
}

\makeatother

%% TODO doing, warning
\newif\ifdebug%
\newcommand{\draft}{\debugtrue}
\newcommand{\final}{\debugfalse}
\newcommand{\todo}[2][FORGOT TO DO SOMETHING]{\ifdebug%
  {%
    \color{red}
    #2}\else \PackageError{}{#1}{#2}#2\fi}%
\newcommand\doing[2][FORGOT TO DO SOMETHING]{\ifdebug%
  {%
    \color{blue}
    #2}\else \PackageError{}{#1}{#2}#2\fi}%
\newcommand\warning[1]{\ifdebug%
  {%
    \color{red}
    #1}\fi}
\newcommand\note[1]{\ifdebug%
  {\pdfcomment[color=orange,opacity=0.4,subject=note]{#1}} \fi}
