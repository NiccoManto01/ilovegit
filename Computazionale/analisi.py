import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def format_with_error(val, err, precision=1):
    """
    Restituisce una stringa tipo 0.054(2) per val=0.054, err=0.002.
    precision: numero di cifre significative dell'errore.
    """
    if err == 0:
        return f"{val:.{precision}f}(0)"
    # Trova la potenza di 10 dell'errore
    exp = int(np.floor(np.log10(abs(err))))
    # Arrotonda l'errore a una cifra significativa
    err_rounded = round(err, -exp + (precision - 1))
    # Trova la cifra tra parentesi
    err_digit = int(err_rounded * 10**(-exp + (precision - 1)))
    # Arrotonda il valore con lo stesso numero di decimali
    val_rounded = round(val, -exp + (precision - 1))
    fmt = f"{{:.{-exp + (precision - 1)}f}}({{}})"
    return fmt.format(val_rounded, err_digit)


def format_with_error_sci(val, err, precision=2):
    """
    Restituisce una stringa tipo 6.67(7) 10^-4 per val=0.000667, err=0.000007.
    precision: numero di cifre significative dell'errore.
    """
    if err == 0:
        return f"{val:.{precision}f}(0)"
    exp = int(np.floor(np.log10(abs(val)))) if val != 0 else 0
    val_scaled = val / 10**exp
    err_scaled = err / 10**exp
    # Trova la potenza di 10 dell'errore scalato
    err_exp = int(np.floor(np.log10(abs(err_scaled)))) if err_scaled != 0 else 0
    dec = max(-err_exp + (precision - 1), 0)
    err_rounded = round(err_scaled, dec)
    err_digit = int(err_rounded * 10**dec)
    val_rounded = round(val_scaled, dec)
    fmt = fr"{{:.{dec}f}}({{}}) $\times 10^{{{{{exp}}}}}$"
    return fmt.format(val_rounded, err_digit)





nfile = ["12","16b","16","20"]
nfileint = [12,16,16,20]
#t0a2int = [35,47,61,78]             # t0 wilson flow calcolato (/0.08)
t0a2int =  [28,38,49,62]
t0a2real = [2.79,3.78,4.87,6.20]    # t0/a^2 wilson flow tabulato

def main1():
    e = 0  # Cambia qui per altri file
    fout_ch = f"Grafici/run1_{nfile[e]}"
    ftxt_ch = f"{fout_ch}_sums.txt"

    tabella = np.loadtxt(ftxt_ch)
    print(tabella.shape)  # (n, m)
    nn = tabella.shape[0] 
    nconfig = tabella.shape[1]
    elementi = [1, int(0.125*t0a2int[e]), int(0.25*t0a2int[e]), int(t0a2int[e])]  # Indici di Wilson flow time
    nbins = 300

    # Crea una figura con 4 sottoplot (2x2)
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()  # Per iterare facilmente

    for i, in_idx in enumerate(elementi):
        if in_idx > nn:
            print(f"Attenzione: indice {in_idx} fuori range (nn={nn})")
            continue
        data = tabella[in_idx-1]
        titolo = f"Somme Q - {nfile[e]} - Wilson flow time {in_idx}"
        axs[i].hist(data, bins=nbins, color='steelblue')
        axs[i].set_title(titolo)
        axs[i].set_xlabel("Topological Charge Q")
        axs[i].set_ylabel("Counts")
        axs[i].grid(True)
        axs[i].set_facecolor('gainsboro')
        axs[i].set_axisbelow(True)
        axs[i].set_xlim([-6, 6])
        axs[i].set_ylim(bottom=0)
    
    # Casella con nn e nconfig
    info = f"nn = {nn}\nnconfig = {nconfig}"
    axs[1].text(0.98, 0.98, info, transform=axs[1].transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    plt.tight_layout()
    plt.savefig(f"{fout_ch}_istogrammi.png")
    plt.close()
    print(f"Salvato istogramma multiplo: {fout_ch}_istogrammi.png")
    


    # Calcolo e plot Q2 per tutti i file in un unico grafico
    colori = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green']
    labels = [str(nfile[i]) for i in range(4)]

    plt.figure(figsize=(8,5))
    for idx in range(4):
        fout_ch = f"Grafici/run1_{nfile[idx]}"
        ftxt_ch = f"{fout_ch}_sums.txt"
        tabella = np.loadtxt(ftxt_ch)
        nn = tabella.shape[0]
        nconfig = tabella.shape[1]
        Volume = nfileint[idx] ** 4
        Q2 = np.zeros(nn)
        for j in range(nn):
            for k in range(nconfig):
                Q2[j] += tabella[j][k] ** 2
            Q2[j] *= (j+1)*(j+1)/nconfig/Volume/100
        plt.plot(range(1, nn+1), Q2, marker='o', color=colori[idx], label=labels[idx])

    plt.title(r"$t^2\ \chi$ vs Wilson flow time $t$ ")
    plt.xlabel("Wilson flow time $t$")
    plt.ylabel(r"$t^2\ \chi$")
    plt.grid(True)
    plt.gca().set_facecolor('gainsboro')
    plt.legend()
    plt.tight_layout()
    plt.savefig("Grafici/Q2_confronto.png")
    plt.close()
    print("Salvato grafico Q2 sovrapposti: Grafici/Q2_confronto.png")



# propago con la funzione f(x) = x*x ogni valore del vettore
# calcolo i valori cluster non propagati xc
# propago i cluster con la funzione f(xc)

def jackknife(data):
    
    n = len(data)
    cluster = np.empty(n)
    media = np.mean(data)
    for i in range(n):
        cluster[i] = media - (data[i] -media)/(n-1)
    media_cluster = np.mean(cluster)
    var_cluster = (n - 1) / n * np.sum((cluster - media_cluster) ** 2)
    errore_cluster = np.sqrt(var_cluster)
    return media, errore_cluster



def main2():

    eQ2 = np.zeros(len(t0a2int))
    Q2 = np.zeros(len(t0a2int))
    t0a2inv = np.zeros(len(t0a2int))
    for i in range(len(t0a2int)):    
        fout_ch = f"Grafici/run1_{nfile[i]}"    # nome del file
        ftxt_ch = f"{fout_ch}_sums_Dani.txt" 
        
        tabella = np.loadtxt(ftxt_ch)   #raccolgo i dati di ogni file
        nn = tabella.shape[0] 
        nconfig = tabella.shape[1]
        
        vettQ2 = np.empty(nconfig)
        Volume = nfileint[i] ** 4
        for k in range(nconfig):        # calcolo la media dei quadrati sulle config
            vettQ2[k] = (tabella[t0a2int[i]][k]**2)*(t0a2real[i]**2)/Volume
            
        (Q2[i],eQ2[i]) = jackknife(vettQ2)   # asse y
        t0a2inv[i] = 1/t0a2real[i]           # asse x



    plt.figure(figsize=(8,5))
    plt.errorbar(t0a2inv, Q2,yerr=eQ2, marker='o', linestyle='', color='black', capsize=4)
    
    # Fit lineare pesato sugli errori
    popt, pcov = np.polyfit(t0a2inv, Q2, 1, w=1/eQ2, cov=True)
    m, q = popt
    m_err, q_err = np.sqrt(np.diag(pcov))

    # Retta di fit
    xfit = np.linspace(0, max(t0a2inv) + 0.02, 100)
    yfit = m * xfit + q
    plt.plot(xfit, yfit, 'r-', label=fr"Fit: $y = {m:.3g}x + {q:.3g}$")

    # Pallino rosso e barra d'errore a x=0
    plt.errorbar(0, q, yerr=q_err, fmt='o', color='red', capsize=4, 
                 label='Fit extrapolation')
    exp_val = 0.000667
    exp_err = 0.000007
    plt.errorbar(0, exp_val, yerr=exp_err, fmt='s', color='blue', capsize=4, 
                 label=fr"Expected: {format_with_error(exp_val, exp_err, 1)}")

    # Calcolo chi quadro
    chi2 = np.sum(((Q2 - (m * t0a2inv + q)) / eQ2) ** 2)
    ndof = len(Q2) - 2

    # Legenda personalizzata
    legend_text = (fr"Fit: $\tilde\chi^2$ = {chi2/ndof:.2f}")
    plt.legend([legend_text,
                "Data [20 - 16 - 16b - 12]",
                fr"Fit extrap: $t_0^2\ \chi_0 =$ {format_with_error_sci(q, q_err, 1)}", 
                fr"Expected: $t_0^2\ \chi_0 =$ {format_with_error_sci(exp_val, exp_err, 1)}"])

    plt.title(r"Extrapolation of $\chi$ in the continuum limit")
    plt.xlabel(r"$\dfrac{a^2}{t_0}$")
    plt.ylabel(r"$t^2_0\ \chi$", rotation=90)
    plt.ylim(top=7.4e-4)
    plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
#    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.grid(True)
    plt.gca().set_facecolor('gainsboro')
    plt.tight_layout()
    plt.savefig(f"Grafici/Acacia.png")
    plt.close()



def jackknife_rapp(data1,data2):
    
    n = len(data1)
    cluster1 = np.empty(n)
    cluster2 = np.empty(n)
    cluster_rapp = np.empty(n)
    media1 = np.mean(data1)
    media2 = np.mean(data2)
    media_rapp = media2/media1
    for i in range(n):
        cluster1[i] = media1 - (data1[i] -media1)/(n-1)
        cluster2[i] = media2 - (data2[i] -media2)/(n-1)
        cluster_rapp[i] = cluster2[i]/cluster1[i]

    varianza = (n - 1) / n * np.sum((cluster_rapp - media_rapp) ** 2)
    errore = np.sqrt(varianza)
    return media_rapp, errore


porzioni = [1.0,0.25,0.5,0.75,1.2,1.4]
colori = ['black','blue','green','yellow','orange','red']
stili = ['.','.','.','.','.','.']


def main3():

    eChi_rapp = np.zeros((len(porzioni),len(t0a2int)))
    Chi_rapp = np.zeros((len(porzioni),len(t0a2int)))

    # calcolo dei quartetti di t^2*Chi ed errori  
    t0a2inv = np.zeros(len(t0a2int))
    for i in range(len(t0a2int)):    

        fout_ch = f"Grafici/run1_{nfile[i]}"    # nome del file
        ftxt_ch = f"{fout_ch}_sums.txt" 
       
        tabella = np.loadtxt(ftxt_ch)   #raccolgo i dati di ogni file
        nn = tabella.shape[0] 
        nconfig = tabella.shape[1]
       
        matrChi = np.empty((len(porzioni),nconfig))
        Volume = nfileint[i] ** 4
        for c in range(len(porzioni)):
            t0a2inv[i] = 1/(t0a2real[i])           # asse x
            for k in range(nconfig):        
                Wriga = int(porzioni[c]*t0a2int[i])
                t2a4real = (t0a2real[i])**2 
                matrChi[c][k] = (tabella[Wriga][k]**2)*t2a4real/Volume
                    
            (Chi_rapp[c][i],eChi_rapp[c][i]) = jackknife_rapp(matrChi[0],matrChi[c])   # assi y

    legend_texts = []
    plt.figure(figsize=(8,5))
    for c in range(1,len(porzioni)):
        
        plt.errorbar(t0a2inv, Chi_rapp[c],yerr=eChi_rapp[c], marker=stili[c], color='black', linestyle='', capsize=4)
    
        # Fit lineare pesato sugli errori
        popt, pcov = np.polyfit(t0a2inv, Chi_rapp[c], 1, w=1/eChi_rapp[c], cov=True)
        m, q = popt
        m_err, q_err = np.sqrt(np.diag(pcov))

        # Retta di fit
        xfit = np.linspace(0, max(t0a2inv) + 0.02, 100)
        yfit = m * xfit + q
        plt.plot(xfit, yfit, label=fr"Fit: $y = {m:.3g}x + {q:.3g}$",color=colori[c])

        # Pallino e barra d'errore a x=0
        plt.errorbar(0, q, yerr=q_err, fmt='.', color=colori[c], capsize=4)

        # Calcolo chi quadro
        chi2 = np.sum(((Chi_rapp[c] - (m * t0a2inv + q)) / eChi_rapp[c]) ** 2)
        ndof = len(Chi_rapp[c]) - 2

        # Legenda personalizzata
        legend_texts.append(
            fr"{porzioni[c]:.2f} $t_0$:" + "\t"
                                    + fr"$q={format_with_error(q,q_err,1)}$" + "\t" 
                                    + fr"$\tilde\chi^2$={chi2/ndof:.2f}"
        )
    
    # Riga nera a 1.00
    plt.hlines(1, xmin=0, xmax=max(t0a2inv) + 0.02, color='black', linestyle=':', linewidth=3)

    plt.legend(legend_texts, loc='best')

    plt.title(r"Topological Susceptibility $\chi$ for different flow times")
    plt.xlabel(r"$\dfrac{a^2}{t_0}$")
    plt.ylabel(r"$\chi_t / \chi$", rotation=90)
    plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
#    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.grid(True)
    plt.gca().set_facecolor('gainsboro')
    plt.tight_layout()
    plt.savefig(f"Grafici/Baobab.png")
    plt.close()

    # --- PUBBLICAZIONE BAOBAB2: fit con q fissato a 1 ---
    legend_texts_fixed = []
    plt.figure(figsize=(8,5))
    for c in range(1, len(porzioni)):

        plt.errorbar(t0a2inv, Chi_rapp[c],yerr=eChi_rapp[c], marker=stili[c], color='black', linestyle='', capsize=4)

        # Fit lineare con intercetta fissata a 1
        w = 1 / eChi_rapp[c]**2
        x = t0a2inv
        y = Chi_rapp[c]
        q_fixed = 1

        num = np.sum(w * x * (y - q_fixed))
        den = np.sum(w * x * x)
        m = num / den
        m_err = np.sqrt(1 / den)

        # Retta di fit
        xfit = np.linspace(0, max(x) + 0.02, 100)
        yfit = m * xfit + q_fixed
        plt.plot(xfit, yfit, linestyle='--', color=colori[c], label=fr"Fit: $y = {m:.3g}x + 1$")

        # Calcolo chi quadro
        chi2 = np.sum(w * (y - (m * x + q_fixed))**2)
        ndof = len(y) - 1  # solo m è libero

        legend_texts_fixed.append(
            fr"{porzioni[c]:.2f} $t_0$:   $\tilde\chi^2$={chi2/ndof:.2f}"
        )

    # Riga nera a 1.00
    plt.hlines(1, xmin=0, xmax=max(t0a2inv) + 0.02, color='black', linestyle=':', linewidth=3)

    plt.legend(legend_texts_fixed, loc='best')
    plt.title(r"Topological Susceptibility $\chi$ for different flow times (q fixed to 1)")
    plt.xlabel(r"$\dfrac{a^2}{t_0}$")
    plt.ylabel(r"$\chi_t / \chi$", rotation=90)
    plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.grid(True)
    plt.gca().set_facecolor('gainsboro')
    plt.tight_layout()
    plt.savefig("Grafici/Baobab2.png")
    plt.close()







def main1b():
    e = 3  # Cambia qui per altri file
    fout_ch = f"Esterni/Nick_Q{nfile[e]}"
    ftxt_ch = f"{fout_ch}.txt"

    tabella = np.loadtxt(ftxt_ch)
    print(tabella.shape)  # (n, m)
    nn = tabella.shape[0] 
    nconfig = tabella.shape[1]
    elementi = [1, int(0.125*t0a2int[e]), int(0.25*t0a2int[e]), int(t0a2int[e])]  # Indici di Wilson flow time
    nbins = 300

    # Crea una figura con 4 sottoplot (2x2)
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()  # Per iterare facilmente

    for i, in_idx in enumerate(elementi):
        if in_idx > nn:
            print(f"Attenzione: indice {in_idx} fuori range (nn={nn})")
            continue
        data = tabella[in_idx-1]
        titolo = f"Somme Q - {nfile[e]} - Wilson flow time {in_idx}"
        axs[i].hist(data, bins=nbins, color='steelblue')
        axs[i].set_title(titolo)
        axs[i].set_xlabel("Topological Charge Q")
        axs[i].set_ylabel("Counts")
        axs[i].grid(True)
        axs[i].set_facecolor('gainsboro')
        axs[i].set_axisbelow(True)
        axs[i].set_xlim([-6, 6])
        axs[i].set_ylim(bottom=0)
    
    # Casella con nn e nconfig
    info = f"nn = {nn}\nnconfig = {nconfig}"
    axs[1].text(0.98, 0.98, info, transform=axs[1].transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    plt.tight_layout()
    plt.savefig(f"{fout_ch}_istogrammi.png")
    plt.close()
    print(f"Salvato istogramma multiplo: {fout_ch}_istogrammi.png")
    


    # Calcolo e plot Q2 per tutti i file in un unico grafico
    colori = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green']
    labels = [str(nfile[i]) for i in range(4)]

    plt.figure(figsize=(8,5))
    for idx in range(4):
        fout_ch = f"Esterni/Nick_Q{nfile[idx]}"
        ftxt_ch = f"{fout_ch}.txt"
        tabella = np.loadtxt(ftxt_ch)
        nn = tabella.shape[0]
        nconfig = tabella.shape[1]
        Volume = nfileint[idx] ** 4
        Q2 = np.zeros(nn)
        for j in range(nn):
            for k in range(nconfig):
                Q2[j] += tabella[j][k] ** 2
            Q2[j] *= (j+1)*(j+1)/nconfig/Volume/100
        plt.plot(range(1, nn+1), Q2, marker='o', color=colori[idx], label=labels[idx])

    plt.title(r"$t^2\ \chi$ vs Wilson flow time $t$ ")
    plt.xlabel("Wilson flow time $t$")
    plt.ylabel(r"$t^2\ \chi$")
    plt.grid(True)
    plt.gca().set_facecolor('gainsboro')
    plt.legend()
    plt.tight_layout()
    plt.savefig("Esterni/Nick_Q2_confronto.png")
    plt.close()
    print("Salvato grafico Q2 sovrapposti: Grafici/Q2_confronto.png")




def main2b():

    Q2 = np.zeros(len(t0a2int))
    eQ2 = np.zeros(len(t0a2int))
    t0a2inv = np.zeros(len(t0a2int))
    for i in range(len(t0a2int)):    
        ftxt_ch = f"Esterni/Nick_Q{nfile[i]}.txt"    # nome del file
        
        tabella = np.loadtxt(ftxt_ch)   #raccolgo i dati di ogni file
        nn = tabella.shape[0] 
        nconfig = tabella.shape[1]
        
        vettQ2 = np.zeros(nconfig)
        Volume = nfileint[i] ** 4
        for k in range(nconfig):        # calcolo i quadrati
            vettQ2[k] = (tabella[t0a2int[i]][k]**2)*(t0a2real[i]**2)/Volume
            
        (Q2[i],eQ2[i]) = jackknife(vettQ2)   # asse y
        t0a2inv[i] = 1/t0a2real[i]           # asse x



    plt.figure(figsize=(8,5))
    plt.errorbar(t0a2inv, Q2,yerr=eQ2, marker='o', linestyle='', color='black', capsize=4)
    
    # Fit lineare pesato sugli errori
    popt, pcov = np.polyfit(t0a2inv, Q2, 1, w=1/eQ2, cov=True)
    m, q = popt
    m_err, q_err = np.sqrt(np.diag(pcov))

    # Retta di fit
    xfit = np.linspace(0, max(t0a2inv) + 0.02, 100)
    yfit = m * xfit + q
    plt.plot(xfit, yfit, 'r-', label=fr"Fit: $y = {m:.3g}x + {q:.3g}$")

    # Pallino rosso e barra d'errore a x=0
    plt.errorbar(0, q, yerr=q_err, fmt='o', color='red', capsize=4, 
                 label='Fit extrapolation')
    exp_val = 0.000667
    exp_err = 0.000007
    plt.errorbar(0, exp_val, yerr=exp_err, fmt='s', color='blue', capsize=4, 
                 label=fr"Expected: {format_with_error(exp_val, exp_err, 1)}")

    # Calcolo chi quadro
    chi2 = np.sum(((Q2 - (m * t0a2inv + q)) / eQ2) ** 2)
    ndof = len(Q2) - 2

    # Legenda personalizzata
    legend_text = (fr"Fit: $\tilde\chi^2$ = {chi2/ndof:.2f}")
    plt.legend([legend_text,
                "Data [20 - 16 - 16b - 12]",
                fr"Fit extrap: $t_0^2\ \chi_0 =$ {format_with_error_sci(q, q_err, 1)}", 
                fr"Expected: $t_0^2\ \chi_0 =$ {format_with_error_sci(exp_val, exp_err, 1)}"])

    plt.title(r"Extrapolation of $\chi$ in the continuum limit")
    plt.xlabel(r"$\dfrac{a^2}{t_0}$")
    plt.ylabel(r"$t^2_0\ \chi$", rotation=90)
    plt.ylim(top=8.4e-4)
    plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
#    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.grid(True)
    plt.gca().set_facecolor('gainsboro')
    plt.tight_layout()
    plt.savefig(f"Esterni/AcaciaNick.png")
    plt.close()




if __name__ == "__main__":
    main2()