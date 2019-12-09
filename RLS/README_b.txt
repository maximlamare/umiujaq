Base de données solnu:

umi-solnu-mns et umi-solnumnt pour 2 périodes (référencé à 2 dates, cf ci-dessous)

- date du 01/10/2017:  concatenation de 4 dates (27/09/2017 ; 28/09/2017 ; 02/10/2017 et 04/10/2017) puis extraction du gridded MNT et MNS

- date du 01/10/2018: concatenation des nuages pours 6 dates (18/09/2018 ; 21/09/2018 ; 24/09/2018 ; 28/09/2018 ; 03/10/2018 ; 09/10/2018 ) puis extraction du gridded MNT et MNS


Base de données data:

+ umi-gridded: MNT sur toute la période avec résolution à 6 cm.

+ height-mnt-mnt: Hauteurs de neige sur toute la période par différence des MNT par rapport au mntsolnu (référence 2017 (27/09/2017 ; 28/09/2017 ; 02/10/2017 et 04/10/2017).  La référence d'automne 2018 n'est pas assez propre pour servir de référence, et si on cat automne 17 et automne 18 pas de gain. Résolution à 6 cm.              

+ height-mns-mnt: Différence MNS - mntsolnu permet d'avoir les hauteurs de végétation pour la période sans neige. Pour la période où toute la végétation est enfouie on doit retrouver mnt - mnt. Pour la période mixte cette différence n'a pas d'intérêt direct. Analyse des 2 saisons à partir de la référence MNT solnu de 2017. Résolution à 6 cm.


