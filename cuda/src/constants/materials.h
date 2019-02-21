#ifndef CONSTANTS_MATERIALS_H
#define CONSTANTS_MATERIALS_H

#include "colours.h"
#include "../material.h"

const Material default_red   (red   , red   , black); //, black, 0, 0.3f, 0, 0, false, 0.0f, 1.0f, false);
const Material default_yellow(yellow, yellow, black); //, black, 0, 0.3f, 0, 0, false, 0.0f, 1.0f, false);
const Material default_green (green , green , black); //, black, 0, 0.3f, 0, 0, false, 0.0f, 1.0f, false);
const Material default_cyan  (cyan  , cyan  , black); //, black, 0, 0.3f, 0, 0, false, 0.0f, 1.0f, false);
const Material default_blue  (blue  , blue  , black); //, black, 0, 0.8f, 0, 0, false, 0.0f, 1.0f, false);
const Material default_purple(purple, purple, black); //, black, 0, 0.8f, 0, 0, false, 0.0f, 1.0f, false);
const Material default_white (white , white , black); //, black, 0, 0.8f, 0, 0, false, 0.0f, 1.0f, false);

const Material m_sol_base03  (sol_base03  , sol_base03  , black );
const Material m_sol_base02  (sol_base02  , sol_base02  , black );
const Material m_sol_base01  (sol_base01  , sol_base01  , black );
const Material m_sol_base00  (sol_base00  , sol_base00  , black );
const Material m_sol_base0   (sol_base0   , sol_base0   , black );
const Material m_sol_base1   (sol_base1   , sol_base1   , black );
const Material m_sol_base2   (sol_base2   , sol_base2   , black );
const Material m_sol_base3   (sol_base3   , sol_base3   , black );
const Material m_sol_yellow  (sol_yellow  , sol_yellow  , black ); //, black, 0, 0.3f, 0, 0, false, 0.0f, 1.0f, false);
const Material m_sol_orange  (sol_orange  , sol_orange  , black ); //, black, 0, 0.3f, 0, 0, false, 0.0f, 1.0f, false);
const Material m_sol_red     (sol_red     , sol_red     , black ); //, black, 0, 0.3f, 0, 0, false, 0.0f, 1.0f, false);
const Material m_sol_magenta (sol_magenta , sol_magenta , black ); //, black, 0, 0.8f, 0, 0, false, 0.0f, 1.0f, false);
const Material m_sol_violet  (sol_violet  , sol_violet  , black ); //, black, 0, 0.3f, 0, 0, false, 0.0f, 1.0f, false);
const Material m_sol_blue    (sol_blue    , sol_blue    , black ); //, black, 0, 0.8f, 0, 0, false, 0.0f, 1.0f, false);
const Material m_sol_cyan    (sol_cyan    , sol_cyan    , black ); //, black, 0, 0.3f, 0, 0, false, 0.0f, 1.0f, false);
const Material m_sol_green   (sol_green   , sol_green   , black ); //, black, 0, 0.3f, 0, 0, false, 0.0f, 1.0f, false);

const Material m_light (black, black, vec3(4.0f, 4.0f, 4.0f));

#endif
