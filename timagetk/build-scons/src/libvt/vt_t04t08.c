
#include <vt_t04t08.h>

#define _FALSE_ 0
#define _TRUE_  1

/*-------------------- statistics --------------------*/
#if defined(_STATISTICS_)
extern int TESTS_nb;
#define GET_VAL(tab,index) ( ++TESTS_nb ? tab[index] : tab[index] )
#else
#define GET_VAL(tab,index) tab[index]
#endif
/*----------------------------------------------------*/



void Compute_T04_and_T08( int neighbors[9], int *t04, int *t08 )
{
  *t04 = Compute_T04( neighbors );
  *t08 = Compute_T08( neighbors );
}

/* Compte les 4-composantes connexes dans un 8-voisinage.

   Les points consideres sont ceux dont la valeur est 
   egale a 0. Le point central n'est pas compte
   (il est considere comme appartenant a l'objet).
   On ne compte que les 4-composantes qui sont 
   4-adjacentes au point central.

RETURN
   Retourne le nombre de composantes connexes.

*/

int Compute_T04( int neighbors[9] )
{
    
    /*--- the neighborhood is numbered as follows 

        0  1  2
        3  4  5
        6  7  8

    ---*/

  int nb_cc = 0;
 
  if ( GET_VAL( neighbors,  1 ) == 0 ) {
    nb_cc ++;
    if ( (GET_VAL( neighbors,  2 ) == 0) && (GET_VAL( neighbors,  5 ) == 0) ) nb_cc --;
  }
  if ( GET_VAL( neighbors,  5 ) == 0 ) {
    nb_cc ++;
    if ( (GET_VAL( neighbors,  8 ) == 0) && (GET_VAL( neighbors,  7 ) == 0) ) nb_cc --;
  }
  if ( GET_VAL( neighbors,  7 ) == 0 ) {
    nb_cc ++;
    if ( (GET_VAL( neighbors,  6 ) == 0) && (GET_VAL( neighbors,  3 ) == 0) ) nb_cc --;
  }
  if ( GET_VAL( neighbors,  3 ) == 0 ) {
    nb_cc ++;
    if ( (GET_VAL( neighbors,  0 ) == 0) && (GET_VAL( neighbors,  1 ) == 0) ) nb_cc --;
  }
  
  /* on a bien le nombre de composantes connexes sauf si 
     tous les points sont a 0,
     auquel cas le resultat est 0 au lieu de 1
  */
  if ( (nb_cc == 0) && (GET_VAL( neighbors,  1 ) == 0) ) nb_cc = 1;

  return(nb_cc);
}

#define _T08_EQUIVALENCE( NEW_LABEL, OLD_CLASS ) { \
        old_label = label_vertex[ OLD_CLASS ];     \
	for ( i=old_label; i<4; i++ ) if ( label_vertex[i] == old_label ) label_vertex[i] = NEW_LABEL; }

int Compute_T08( int neighbors[9] )
{
    
    /*--- the neighborhood is numbered as follows 

        0  1  2       0 . 1
        3  4  5    => . . .
        6  7  8       2 . 3

    ---*/
  int label_vertex[4], label_exists[4];
  register int i, nb, old_label;

  /* nb de 4-voisins != 0 */
  nb = 0;
  if ( GET_VAL( neighbors,  1 ) != 0 ) nb++;
  if ( GET_VAL( neighbors,  3 ) != 0 ) nb++;
  if ( GET_VAL( neighbors,  5 ) != 0 ) nb++;
  if ( GET_VAL( neighbors,  7 ) != 0 ) nb++;
  if ( nb >= 3 ) return ( 1 );
  

  for ( i = 0; i < 4; i ++ ) {
	label_vertex[i] = i;
	label_exists[i] = _FALSE_;
    }


  /* */

  if ( GET_VAL( neighbors, 1 ) != 0 ) {
    label_vertex[1] = 0;
    label_exists[0] = label_exists[1] = _TRUE_;
  }
  if ( GET_VAL( neighbors, 3 ) != 0 ) {
    label_vertex[2] = 0;
    label_exists[0] = label_exists[2] = _TRUE_;
  }

  if ( GET_VAL( neighbors, 5 ) != 0 ) {
    if ( label_vertex[1] < label_vertex[3] ) {      _T08_EQUIVALENCE( label_vertex[1], 3 ) }
    else if ( label_vertex[3] < label_vertex[1] ) { _T08_EQUIVALENCE( label_vertex[3], 1 ) }
    label_exists[1] = label_exists[3] = _TRUE_;
  }

  if ( GET_VAL( neighbors, 7 ) != 0 ) {
    if ( label_vertex[2] < label_vertex[3] ) {      _T08_EQUIVALENCE( label_vertex[2], 3 ) }
    else if ( label_vertex[3] < label_vertex[2] ) { _T08_EQUIVALENCE( label_vertex[3], 2 ) }
    label_exists[2] = label_exists[3] = _TRUE_;
  }
  
  nb = 0;

  if ( label_exists[0] == _TRUE_ ) {         nb ++; }
  else { if ( GET_VAL( neighbors, 0 ) != 0 ) nb ++; }

  if ( label_vertex[1] == 1 ) {
    if ( label_exists[1] == _TRUE_ ) {         nb ++; }
    else { if ( GET_VAL( neighbors, 2 ) != 0 ) nb ++; }
  }

  if ( label_vertex[2] == 2 ) {
    if ( label_exists[2] == _TRUE_ ) {         nb ++; }
    else { if ( GET_VAL( neighbors, 6 ) != 0 ) nb ++; }
  }
  
  if ( label_vertex[3] == 3 ) {
    if ( label_exists[3] == _TRUE_ ) {         nb ++; }
    else { if ( GET_VAL( neighbors, 8 ) != 0 ) nb ++; }
  }
  
  return( nb );
}


